import os
import logging
import pandas as pd
import pywikibot
from pywikibot import Site
from pywikibot.data import api
from datetime import datetime
import spacy
from ..kg import kg_dataclasses as kg
from .. import hlp_functions as hlp

log_file_path = os.path.join(os.getcwd(), 'kg_builder.log')

logger = logging.getLogger(__name__)
logging.basicConfig(filename=log_file_path, encoding='utf-8', level=logging.DEBUG)

# ## Define global variables
# ## -----------------------

# Get type lookups from Wikidata reference info
rebel_flair_overview, _, _, _, _ = hlp.get_wikidata_prepared_info('reference_info/wikidata_references.pkl')

subjects = rebel_flair_overview.copy()
subjects = subjects[['subject', 'subject NER mapping']].dropna()
subject_type_lookup = list(zip(subjects['subject'], subjects['subject NER mapping']))
subject_type_lookup = dict([(subject, mapping) for subject, mapping in subject_type_lookup if ',' not in mapping])

objects = rebel_flair_overview.copy()
objects = objects[['object', 'object NER mapping']].dropna()
object_type_lookup = list(zip(objects['object'], objects['object NER mapping']))
object_type_lookup = dict([(objectt, mapping) for objectt, mapping in object_type_lookup if ',' not in mapping])

TYPE_LOOKUPS = subject_type_lookup | object_type_lookup

main_relations = hlp.make_lookup_dict_from_df(rebel_flair_overview, 'rebel description', 'wikidata property')
inverse_relations = hlp.make_lookup_dict_from_df(rebel_flair_overview, 'inverse description', 'inverse property or label')

RELATION_IDS = main_relations | inverse_relations


# ## Define functions for knowledge processing
# ## -----------------------------------------


def setup_el_tagger():
    '''
        Sets up the entity linking tagger for OpenTapioca.
    '''
    el_tagger = spacy.blank("en")
    el_tagger.add_pipe('opentapioca')
    return el_tagger


def get_relation_span(relation, article):
    '''
    Takes in an article Relation instance and an Article instance and returns the start and end
    of the relevant sentence(s) in the text from whence the relation comes.
    '''
    # Get the start of the actual relation
    relation_start = min([relation.head_start_char, relation.head_end_char, relation.tail_start_char, relation.tail_end_char])
    # And then the sentence span in which that start occurs
    reference_start = article.sentence_indices[hlp.sentence_in_range(article.sentence_indices, relation_start)]
    # Get the end of the actual relation
    relation_end = max([relation.head_start_char, relation.head_end_char, relation.tail_start_char, relation.tail_end_char])
    # And then the sentence span in which that end occurs
    reference_end = article.sentence_indices[hlp.sentence_in_range(article.sentence_indices, relation_end)]
    # And finally get the start and end of the entire reference
    return (min(reference_start), max(reference_end))


def update_entity_id_for_relations(kg_instance: kg.KGData, current_time):
    '''
    The entities_tracker is of type kg.ChangeLogDict and keeps track of changes made to the entities_tracker:
    when a new entity_id is assigned to a key (composed of a tuple of the lower entity_name and lower
    entity_type) it will be picked up and logged.
    '''
    changes_required = [change for change in kg_instance.entities_tracker.changes if change['datetime'] > current_time]
    if len(changes_required) > 0:
        for change in changes_required:
            logger.info(f'''    update: previous = {change['previous']}; new_value = {change['new_value']}''')
            kg_instance.entities = [entity for entity in kg_instance.entities if entity.EntityId != change['previous']]
            heads_to_update = [i for i, relation in enumerate(kg_instance.relations) if relation.HeadId == change['previous']]
            for i in heads_to_update:
                kg_instance.relations[i].HeadId = change['new_value']
            tails_to_update = [i for i, relation in enumerate(kg_instance.relations) if relation.TailId == change['previous']]
            for i in tails_to_update:
                kg_instance.relations[i].TailId = change['new_value']
                
                
def entity_in_tracker(kg_instance: kg.KGData, article_entity_text: str, article_entity_type: str) -> (str, int):
    '''
    Test if an article entity is already in the KG entities_tracker. Example outout:
    ('rh2diqiMB', 3) where 'rh2diqiMB' is the entity_id found and 3 means it is the 
    fourth entity in the list of entities.
    '''
    entity_key = (article_entity_text.lower(), article_entity_type.lower())
    if entity_key in kg_instance.entities_tracker.keys():
        entity_id = kg_instance.entities_tracker[entity_key]
        entity_i = [i for i, entity in enumerate(kg_instance.entities) if entity.EntityId == entity_id][0]
        return entity_id , entity_i
    return None, None


def get_direct_entity_match(entity_name: str) -> bool:
    '''
    Takes in an entity name and tries to find a match in Wikidata. Exact matches are accepted.
    If partial or multiple matches are returned then True (implying it is worth using EL to see
    if a good match can be confirmed) otherwise False (implying it is not worth looking for
    further data in Wikidata).
    '''
    site = Site('wikidata', 'wikidata')
    repo = site.data_repository()

    parameters = {'action': 'wbsearchentities',
                  'search': entity_name,
                  'language': 'en',
                  'type': 'item'}
    request = api.Request(site=site, parameters = parameters)
    # Search results include full and partial matches on the label and/or aliases
    search_results = request.submit()

    entity_matches = []
    
    # If there is more than one matching search result...
    if len(search_results['search']) > 0:
        for item in search_results['search']: 
            property_id = item['id']
            label = item.get('label')
            # NOTE only matching aliases are returned, not all aliases
            aliases = item.get('aliases')
            # Check if the entity_name and the label are an exact match
            if entity_name.lower() == label.lower():
                entity_matches.append(property_id)
            # Check if the entity and any of the returned aliases are an exact match
            if aliases is not None:
                aliases = [alias.lower() for alias in aliases]
                if entity_name.lower() in aliases:
                    entity_matches.append(property_id)
    # If there were no matching search results at all then it's probably not worth running EL
    if len(search_results['search']) == 0:
        return False
    # If there was only one match we don't *need* to run EL because we have an answer
    elif len(entity_matches) == 1:
        return entity_matches[0]
    # Otherwise there was more than one match or a partial match in which case 
    # it is worth running EL to see if we can disambiguate
    else: 
        return True
    
    
def get_new_aliases(article: kg.Article, article_entity_text: str, kg_also_known: set) -> str:
    '''
    Check if there is an alias associated with the entity and whether or not it has
    already been noted. If it is a new alias append.
    '''
    alt_name = hlp.get_key_for_value(article.alt_names, article_entity_text)
    new_also_knowns = set()
    if alt_name is not None and alt_name not in kg_also_known:
        new_also_knowns.add(alt_name)
    return new_also_knowns


def get_first_mention_sentence(article, article_entity):
    '''
    Get the sentence in which an entity is first mentioned in the given article. 
    Used for entity linking where the assumption is that the first time an entity
    is introduced may be where it is introduced most unambiguously in the article
    for the sake of the reader.
    '''
    mention_start = article.article_text.find(article_entity)
    first_mention = None
    if mention_start != -1:
        start, stop = article.sentence_indices[hlp.sentence_in_range(article.sentence_indices, mention_start)]
        first_mention = article.article_text[start:stop]
    return first_mention


def swap_out_obsolete_entity(kg_instance, old_new_pair):
    '''
    Takes in a previous value and new value entity id and makes sure that the relations are updated.
    '''
    previous, new_value = old_new_pair
    logger.info(f'''    swap_out: previous = {previous}; new_value = {new_value}''')
    # Added this line when transferring to el_enhancements - check if really needed and correct?
    kg_instance.entities = [entity for entity in kg_instance.entities if entity.EntityId != previous]
    # Update any head entity_id's that need updating
    heads_to_update = [i for i, relation in enumerate(kg_instance.relations) if relation.HeadId == previous]
    for i in heads_to_update:
        kg_instance.relations[i].HeadId = new_value
    # Update any tail entity_id's that need updating
    tails_to_update = [i for i, relation in enumerate(kg_instance.relations) if relation.TailId == previous]
    for i in tails_to_update:
        kg_instance.relations[i].TailId = new_value  
        
        
def determine_switching(kg_instance, also_knowns, entity_type):
    '''
    Determines whether we should switch to editing an existing entity because it matches
    an existing alias already in the KG entities.
    '''
    alias_keys = [(alias.lower(), entity_type.lower()) for alias in also_knowns]
    matching_values = [kg_instance.entities_tracker[key] for key in alias_keys if key in kg_instance.entities_tracker.keys()]
    ex_entity_id = None
    ex_entity_i = None
    if len(matching_values) > 0:
        ex_entity_id = matching_values[0]
        ex_entity_i = [i for i, entity in enumerate(kg_instance.entities) if entity.EntityId == ex_entity_id][0]
        switcheroo = True
    else:
        switcheroo = False
    return switcheroo, ex_entity_id, ex_entity_i


def get_el_status(kg_wdid, kg_wd_pywiki, kg_wd_retries):
    '''
    Get EL status: whether to try OpenTapioca, Pywikibot, or stop trying.
    '''
    el_status = 'el_closed'
    if kg_wdid is None and type(kg_wd_pywiki) == str and 5 < kg_wd_retries < 99:
        el_status = 'do_pywiki'
    elif kg_wdid is None and (kg_wd_pywiki == True or type(kg_wd_pywiki) == str) and 0 <= kg_wd_retries <= 5:
        el_status = 'do_opentapioca'
    return el_status


def update_entities_tracker(kg_instance, name, also_knowns, entity_type, entity_id):
    '''
    Update the entities tracker, and then also make sure that all changes logged are 
    replicated to the relations as well.
    '''
    current_time = datetime.now()
    kg_instance.entities_tracker[(name.lower(), entity_type.lower())] = entity_id
    for alias in also_knowns:
        kg_instance.entities_tracker[(alias.lower(), entity_type.lower())] = entity_id
        # If any entity_id's were updated we need to make sure they are also updated in the relations
        update_entity_id_for_relations(kg_instance, current_time)
        
        
def get_pywikibot_entity_info(wd_wdid: str, article_entity_type: str) -> tuple:
    '''
    Takes in a Wikidata item Id and returns the following data points:
    - WDId
    - label (the main Wikidata name)
    - aliases (list of aliases available on Wikidata)
    - instance_type (the P31 property associated with the WDId on Wikidata)
    - description (the main Wikidata description)
    - WDUrl (the WDUrl associated with the WDId)
    '''
    wd_source = 'PyWikibot'
    site = Site('wikidata', 'wikidata')
    repo = site.data_repository()
    
    item = pywikibot.ItemPage(repo, wd_wdid)
    
    # Get the label (main name), aliases and Wikidata URL
    wd_entity_name = item.labels.get('en', None)
    wd_also_knowns = item.aliases.get('en', None)
    if wd_also_knowns is not None:
        wd_also_knowns = set(wd_also_knowns)
    else:
        wd_also_knowns = set()
    if 'en' in item.text.get('descriptions'):
        wd_description = item.text['descriptions']['en']
    else:
        wd_description = None
    wd_wdurl = 'https://www.wikidata.org/wiki/' + wd_wdid
    
    # There should usually be an instance property (P31) and if it is one of our
    # known lookups we can add it, otherwise we are going to leave the type as-is
    wd_entity_type = article_entity_type
    if item.claims:
        if 'P31' in item.claims:
            ITId = item.claims['P31'][0].getTarget()
            instance = pywikibot.ItemPage(repo, ITId.id)
            wd_entity_type = TYPE_LOOKUPS.get(instance.labels.get('en', None))
            
    valid_el_result = True if wd_wdid is not None and wd_entity_name is not None else False
    
    if valid_el_result:
        wd_entity_type = wd_entity_type if wd_entity_type is not None else article_entity_type
    # Note the placeholder for instance_type is None. The instance_type that comes from
    # the NER process is much more trustwory so we ignore this one in favour of NER's output
        return wd_wdid, wd_entity_name, wd_also_knowns, wd_entity_type, wd_description, wd_wdurl, wd_source
    else:
        return False
    
    
def get_opentapioca_entity_info(article_entity_text: str, article_entity_type, sentence: str, el_tagger):
    '''
    Takes in an entity and a sentence and attempts entity linking. The expected
    el_tagger is spaCy OpenTapioca.
    '''
    wd_source = 'OpenTapioca'
    result = None
    doc = el_tagger(sentence)
    # If we got at least one entity back
    if len(doc.ents) > 0:
        for span in doc.ents:
            # Check if the article_entity_text is in the aliases (which currently includes
            # the label as well) and that the score is better than the last result recorded
            # (if there was one) and update the result accordingly
            if article_entity_text in span._.aliases:
                if result is None:
                    result = (span._.score, span.kb_id_, span._.label, span._.aliases, article_entity_type, span._.description, "https://www.wikidata.org/entity/" + span.kb_id_, wd_source)
                elif span._.score > result[0]:
                    result = (span._.score, span.kb_id_, span._.label, span._.aliases, article_entity_type, span._.description, "https://www.wikidata.org/entity/" + span.kb_id_, wd_source)
    # If we got a result then assemble and return
    if result is not None:
        # Get all the items except the score
        wd_wdid, wd_entity_name, wd_also_knowns, wd_entity_type, wd_description, wd_wdurl, wd_source = result[1:]
        # Remove the label from the list of aliases
        wd_also_knowns = set([alias for alias in wd_also_knowns if alias != wd_entity_name])
        # Get the first description out of the list
        if wd_description is not None:
            if len(wd_description) > 0:
                wd_description = wd_description[0]
        else:
            wd_description = None
            
        valid_el_result = True if wd_wdid is not None and wd_entity_name is not None else False
        
        if valid_el_result:
            # Note the placeholder for entity type. The entity type that comes from the NER 
            # process is much more trustwory so we ignore this one in favour of NER's output
            
            # wd_source = 'PyWikibot'
            site = Site('wikidata', 'wikidata')
            repo = site.data_repository()

            item = pywikibot.ItemPage(repo, wd_wdid)
            
            try:
                if item.claims:
                    if 'P31' in item.claims:
                        ITId = item.claims['P31'][0].getTarget()
                        instance = pywikibot.ItemPage(repo, ITId.id)
                        wd_entity_type = TYPE_LOOKUPS.get(instance.labels.get('en', None))
            except:
                logger.info('    WD redirect scenario - skipping')
            wd_entity_type = wd_entity_type if wd_entity_type is not None else article_entity_type
            return wd_wdid, wd_entity_name, wd_also_knowns, wd_entity_type, wd_description, wd_wdurl, wd_source
    else:
        return False


def process_entity(kg_instance: kg.KGData, 
                     article: kg.Article, 
                     article_entity_text: str, 
                     article_entity_type: str,
                     el_tagger):
    '''
    Algorithm for determining whether to create a new entity, update and existing one,
    or switch to editing an alternate entity and merge data points.
    '''
    
# ------------------ INITIALIZING THE KG ENTITY DATA POINTS --------------------- #

    entity_id, entity_i = entity_in_tracker(kg_instance, article_entity_text, article_entity_type)

    if entity_id is None:

        create_track = True
        logger.info('    On Create Track')

        kg_entity_id = hlp.generate_uid()
        kg_entity_name = article_entity_text
        kg_entity_type = article_entity_type
        kg_wdid = None
        kg_wdsource = None
        kg_wdurl = None
        kg_description = None
        kg_also_knowns = set()
        kg_wd_articles = set()
        kg_wd_pywiki = get_direct_entity_match(article_entity_text)
        kg_wd_retries = 0 if kg_wd_pywiki != False else 99

    else:

        create_track = False
        logger.info('    On Update Track')

        kg_entity_id = kg_instance.entities[entity_i].EntityId
        kg_entity_name = kg_instance.entities[entity_i].Name
        kg_entity_type = kg_instance.entities[entity_i].Type
        kg_wdid = kg_instance.entities[entity_i].WDId
        kg_wdsource = kg_instance.entities[entity_i].WDSource
        kg_wdurl = kg_instance.entities[entity_i].WDUrl
        kg_description = kg_instance.entities[entity_i].WDDescription
        kg_also_knowns = kg_instance.entities[entity_i].AlsoKnownAs
        kg_wd_articles = kg_instance.entities[entity_i].WDRetryArticles
        kg_wd_pywiki = kg_instance.entities[entity_i].PywikiStatus
        kg_wd_retries = kg_instance.entities[entity_i].WDRetry


# ------------------ GATHERING INFO & DOING ANY SWITCHES REQUIRED --------------------- #
        
    # 1. Make all the statuses False to start with
    also_known_status = new_also_known_switch = el_results = wd_id_exists_switch = wd_also_known_switch = False

    # 2. Get new also knowns and also known status
    new_also_knowns = get_new_aliases(article, article_entity_text, kg_also_knowns)
    if len(new_also_knowns) > 0:    
        also_known_status = True

        # 3. Check if we should switch based on the new_also_knowns
        new_also_known_switch, existing_entity_id, existing_entity_i = determine_switching(kg_instance, new_also_knowns, kg_entity_type)

        if new_also_known_switch:

            # SCENARIO S-1
            # Switching to another entity as existing also knowns found

            existing_also_knowns = kg_instance.entities[existing_entity_i].AlsoKnownAs
            kg_instance.entities[existing_entity_i].Name = article_entity_text
            kg_instance.entities[existing_entity_i].AlsoKnownAs = existing_also_knowns | new_also_knowns

            update_entities_tracker(kg_instance, 
                                    kg_instance.entities[existing_entity_i].Name,
                                    kg_instance.entities[existing_entity_i].AlsoKnownAs, 
                                    kg_instance.entities[existing_entity_i].Type, 
                                    kg_instance.entities[existing_entity_i].EntityId)
            

            logger.info('    process_entity SCENARIO S-1')

            return existing_entity_id, kg_instance.entities[existing_entity_i].Name

    # 4. Determine if EL will be done and which variety
    el_status = get_el_status(kg_wdid, kg_wd_pywiki, kg_wd_retries)

    # 5. Then do EL

    if el_status == 'do_pywiki':

        kg_wd_retries = 99

        el_results = get_pywikibot_entity_info(kg_wd_pywiki, article_entity_type)
        if el_results != False:
            wd_wdid, wd_entity_name, wd_also_knowns, wd_entity_type, wd_description, wd_wdurl, wd_source = el_results

    if el_status == 'do_opentapioca' and article.article_id not in kg_wd_articles:

        first_mention = get_first_mention_sentence(article, article_entity_text)
        if first_mention is not None:

            kg_wd_retries += 1
            kg_wd_articles.add(article.article_id)
            
            el_results = get_opentapioca_entity_info(article_entity_text, article_entity_type, first_mention, el_tagger)
            if el_results != False:
                wd_wdid, wd_entity_name, wd_also_knowns, wd_entity_type, wd_description, wd_wdurl, wd_source = el_results

    # 6. Now check if we should switch based in WDId in tracker and get deets if so            
    if el_results != False and wd_wdid in kg_instance.wd_tracker.keys():
        wd_id_exists_switch = True

        if wd_id_exists_switch:

            # SCENARIO S-2
            # Switching to another entity as WDId already in wd_tracker
            existing_entity_id = kg_instance.wd_tracker[wd_wdid]['EntityId']
            existing_entity_i = [i for i, entity in enumerate(kg_instance.entities) if entity.EntityId == existing_entity_id][0]

            existing_also_knowns = kg_instance.entities[existing_entity_i].AlsoKnownAs
            if article_entity_text != kg_instance.entities[existing_entity_i].Name:
                new_also_knowns.add(article_entity_text)
            kg_instance.entities[existing_entity_i].AlsoKnownAs = existing_also_knowns | new_also_knowns

            update_entities_tracker(kg_instance, 
                                    kg_instance.entities[existing_entity_i].Name,
                                    kg_instance.entities[existing_entity_i].AlsoKnownAs, 
                                    kg_instance.entities[existing_entity_i].Type, 
                                    kg_instance.entities[existing_entity_i].EntityId)
            
            
            if not create_track and kg_entity_id != existing_entity_id:
                swap_out_obsolete_entity(kg_instance, (kg_entity_id, existing_entity_id))
                kg_instance.entities_tracker = kg.ChangeLogDict({k: v for k, v in kg_instance.entities_tracker.items() if v != kg_entity_id})

            logger.info('    process_entity SCENARIO S-2')

            return existing_entity_id, kg_instance.entities[existing_entity_i].Name

    elif el_results != False and wd_wdid not in kg_instance.wd_tracker.keys():

    # Check if we should switch based on wdid alias in entities tracker and get deets if so
        wd_also_known_switch, existing_entity_id, existing_entity_i = determine_switching(kg_instance, wd_also_knowns, wd_entity_type)
        if wd_also_known_switch:

            # SCENARIO S-3
            # Switching to another entity as wd_also_knowns in entities tracker

            existing_also_knowns = kg_instance.entities[existing_entity_i].AlsoKnownAs
            # if article_entity_text != kg_instance.entities[existing_entity_i].Name:
            #     new_also_knowns.add(article_entity_text)

            kg_instance.entities[existing_entity_i].Name = wd_entity_name
            kg_instance.entities[existing_entity_i].Type = wd_entity_type
            kg_instance.entities[existing_entity_i].WDId = wd_wdid
            kg_instance.entities[existing_entity_i].WDSource = wd_source
            kg_instance.entities[existing_entity_i].WDUrl = wd_wdurl
            kg_instance.entities[existing_entity_i].WDDescription = wd_description
            kg_instance.entities[existing_entity_i].WDRetry = 99
            kg_instance.entities[existing_entity_i].AlsoKnownAs = existing_also_knowns | new_also_knowns | wd_also_knowns
            kg_instance.entities[existing_entity_i].PywikiStatus = wd_wdid
            kg_instance.entities[existing_entity_i].Updated = datetime.now()

            update_entities_tracker(kg_instance, 
                                    kg_instance.entities[existing_entity_i].Name,
                                    kg_instance.entities[existing_entity_i].AlsoKnownAs, 
                                    kg_instance.entities[existing_entity_i].Type, 
                                    existing_entity_id)
            
            if not create_track and kg_entity_id != existing_entity_id:
                print('This evaluated to TRUE!')
                swap_out_obsolete_entity(kg_instance, (kg_entity_id, existing_entity_id))
                kg_instance.entities_tracker = kg.ChangeLogDict({k: v for k, v in kg_instance.entities_tracker.items() if v != kg_entity_id})

            kg_instance.wd_tracker[wd_wdid] = {'WDUrl': wd_wdurl,
                                               'WDName': wd_entity_name,
                                               'WDType': wd_entity_type,
                                               'WDAlsoKnown': wd_also_knowns,
                                               'WDDescription': wd_description,
                                               'WDSource': wd_source,
                                               'EntityId': existing_entity_id}

            logger.info('    process_entity SCENARIO S-3')

            return existing_entity_id, wd_entity_name


    # ------------------ CREATING NEW ENTITIES --------------------- #

    # SCENARIO C-1
    # A new entity is created where entity linking was not possible
    if create_track and not el_results:
        new_entity = kg.KGEntity(EntityId = kg_entity_id, 
                                 Name = kg_entity_name,
                                 Type = kg_entity_type,
                                 WDRetry = kg_wd_retries, # may have been incremented by 1 or set to 99 depending
                                 AlsoKnownAs = kg_also_knowns | new_also_knowns,
                                 PywikiStatus = kg_wd_pywiki,
                                 Updated = datetime.now())
        kg_instance.entities.append(new_entity)

        update_entities_tracker(kg_instance, 
                                kg_entity_name,
                                kg_also_knowns | new_also_knowns, 
                                kg_entity_type, 
                                kg_entity_id)

        logger.info('    process_entity SCENARIO C-1')

        return kg_entity_id, kg_entity_name

    # SCENARIO C-2
    # a new entity is created where entity linking was possible
    elif create_track and el_results != False and not wd_id_exists_switch and not wd_also_known_switch:
        new_entity = kg.KGEntity(EntityId = kg_entity_id, 
                                 Name = wd_entity_name,
                                 Type = wd_entity_type,
                                 WDId = wd_wdid,
                                 WDUrl = wd_wdurl,
                                 WDDescription = wd_description,
                                 WDRetry = 99,
                                 WDSource = wd_source,
                                 WDRetryArticles = kg_wd_articles,
                                 AlsoKnownAs = kg_also_knowns | new_also_knowns | wd_also_knowns,
                                 PywikiStatus = kg_wd_pywiki,
                                 Updated = datetime.now())
        kg_instance.entities.append(new_entity)

        update_entities_tracker(kg_instance, 
                                wd_entity_name,
                                kg_also_knowns | new_also_knowns | wd_also_knowns, 
                                wd_entity_type, 
                                kg_entity_id)
        
        kg_instance.wd_tracker[wd_wdid] = {'WDUrl': wd_wdurl,
                                           'WDName': wd_entity_name,
                                           'WDType': wd_entity_type,
                                           'WDAlsoKnown': wd_also_knowns,
                                           'WDDescription': wd_description,
                                           'WDSource': wd_source,
                                           'EntityId': kg_entity_id}

        logger.info('    process_entity SCENARIO C-2')

        return kg_entity_id, wd_entity_name


    # ------------------ UPDATING EXISTING ENTITIES --------------------- #

    # SCENARIO U-1
    # Update just the new also knowns into existing entity
    elif not create_track and also_known_status and (el_status == 'el_closed' or el_results == False):

        kg_instance.entities[entity_i].AlsoKnownAs = kg_also_knowns | new_also_knowns

        update_entities_tracker(kg_instance, 
                                kg_entity_name,
                                kg_also_knowns | new_also_knowns, 
                                kg_entity_type, 
                                kg_entity_id)

        logger.info('    process_entity SCENARIO U-1')

        return kg_entity_id, kg_entity_name


    # SCENARIO U-2
    # Update the entity with all the new WD info
    elif not create_track and el_results != False and not wd_id_exists_switch and not wd_also_known_switch:

        kg_instance.entities[entity_i].Name = wd_entity_name
        kg_instance.entities[entity_i].Type = wd_entity_type
        kg_instance.entities[entity_i].WDId = wd_wdid
        kg_instance.entities[entity_i].WDSource = wd_source
        kg_instance.entities[entity_i].WDUrl = wd_wdurl
        kg_instance.entities[entity_i].WDDescription = wd_description
        kg_instance.entities[entity_i].WDRetry = 99
        kg_instance.entities[entity_i].AlsoKnownAs = kg_also_knowns | new_also_knowns | wd_also_knowns
        kg_instance.entities[entity_i].PywikiStatus = wd_wdid
        kg_instance.entities[entity_i].Updated = datetime.now()

        update_entities_tracker(kg_instance, 
                                kg_instance.entities[entity_i].Name,
                                kg_instance.entities[entity_i].AlsoKnownAs, 
                                kg_instance.entities[entity_i].Type, 
                                entity_id)

        kg_instance.wd_tracker[wd_wdid] = {'WDUrl': wd_wdurl,
                                           'WDName': wd_entity_name,
                                           'WDType': wd_entity_type,
                                           'WDAlsoKnown': wd_also_knowns,
                                           'WDDescription': wd_description,
                                           'WDSource': wd_source,
                                           'EntityId': entity_id}

        logger.info('    process_entity SCENARIO U-2')  

        return kg_entity_id, wd_entity_name

    # SCENARIO O-1
    # No changes were required to the entity, with the possible exception of kg_wd_retries
    elif not create_track and not also_known_status and el_results == False:
        kg_instance.entities[entity_i].WDRetry = kg_wd_retries

        logger.info('    process_entity SCENARIO O-1')

        return kg_entity_id, kg_entity_name

    # SCENARIO O-2
    # Ones I may not have thought of
    else:
        logger.info(f'    {create_track, also_known_status, new_also_known_switch, el_results, wd_id_exists_switch, wd_also_known_switch}')
        logger.info('    process_entity SCENARIO O-2')

        return None
    

def update_kg_from_article(kg_instance: kg.KGData, article:kg.Article, el_tagger):
    '''
    Algorithm for processing one article into a new / existing KG.
    '''
    logger.info(f'''Starting: {article.article_id}''')
    # Create an entity to hold the article if it does not yet exist
    if (article.permatitle, 'article') not in kg_instance.entities_tracker:
        article_entity = kg.KGEntity(EntityId = article.article_id, 
                                     Name = article.permatitle,
                                     Type = 'ARTICLE',
                                     WDRetry = 99,
                                     Updated = datetime.now())

        kg_instance.entities.append(article_entity)
        kg_instance.entities_tracker[article.permatitle, 'article'] = article.article_id
    
    # Add / update the entities and relations for the article
    unique_entity_ids = set()
    for relation in article.relations:
        
        if not all([relation.clean_head_text, relation.head_type, relation.clean_tail_text, relation.tail_type]):
            logger.info(f'''    Moving on - relation lacks sufficient info: {relation.clean_head_text, relation.head_type, relation.clean_tail_text, relation.tail_type}''')
            continue
        # First process any new or updated entities and get the entity keys for further processing
        # print(f'''Head relation to be processed: {relation.clean_head_text, relation.head_type}''')
        logger.info(f'''    Head relation to be processed: {relation.clean_head_text, relation.head_type}''')
        (head_entity_id, head_entity_text) = process_entity(kg_instance, article, relation.clean_head_text, relation.head_type, el_tagger)
        logger.info(f'''    Ouptut head_entity_id: {head_entity_id}''')
        unique_entity_ids.add((head_entity_id, head_entity_text))
        logger.info(f'''    Tail relation to be processed: {relation.clean_tail_text, relation.tail_type}''')
        (tail_entity_id, tail_entity_text) = process_entity(kg_instance, article, relation.clean_tail_text, relation.tail_type, el_tagger)
        logger.info(f'''    Output tail_entity_id {tail_entity_id}''')
        unique_entity_ids.add((tail_entity_id, tail_entity_text))
        
        # Now we check if the relation has appeared in the unique list of relations before
        relation_id = kg_instance.relations_tracker.get((head_entity_id, 
                                                         relation.relation_type, 
                                                         tail_entity_id), None)
        
        # If it's a new relation then we add it to both the relations and the tracker
        relation_span = get_relation_span(relation, article)
        if relation_id is None:
            new_relation = kg.KGRelation(RelationId = hlp.generate_uid(), 
                                         HeadId = head_entity_id, 
                                         HeadName = relation.clean_head_text, 
                                         Type = relation.relation_type, 
                                         WDType = RELATION_IDS.get(relation.relation_type),
                                         TailId = tail_entity_id, 
                                         TailName = relation.clean_tail_text,  
                                         Instances = {(article.article_id, 
                                                       article.permatitle,
                                                       relation_span): 1},
                                         Updated = datetime.now())
            kg_instance.relations.append(new_relation)
            kg_instance.relations_tracker[(head_entity_id, 
                                           relation.relation_type, 
                                           tail_entity_id)] = new_relation.RelationId
        
        # It it's an existing relation then we get the existing relation details
        else:
            i, existing_relation = kg_instance.get_relation_by_id(relation_id)
            if existing_relation.Instances.get((article.article_id, 
                                                article.permatitle,
                                                relation_span)) is not None:
                kg_instance.relations[i].Instances[(article.article_id, 
                                                    article.permatitle,
                                                    relation_span)] +=1
                kg_instance.relations[i].Updated = datetime.now()
            else:
                kg_instance.relations[i].Instances[(article.article_id, 
                                                    article.permatitle,
                                                    relation_span)] = 1
                kg_instance.relations[i].Updated = datetime.now()
                
    # And finally create the entity / article relations
    for entity_id, entity_text in unique_entity_ids:
        relation_id = kg_instance.relations_tracker.get((entity_id, 
                                                         'mentioned_in', 
                                                         article.permatitle, 
                                                         'article'), None)
        if relation_id is None:
            relation_id = hlp.generate_uid()
            article_relation = kg.KGRelation(RelationId = relation_id, 
                                             HeadId = entity_id, 
                                             HeadName = entity_text, 
                                             Type = 'mentioned_in', 
                                             TailId = article.article_id, 
                                             TailName = article.permatitle,  
                                             Instances = {('', 
                                                           '',
                                                           ''): article.article_text.lower().count(entity_text.lower())},
                                             Updated = datetime.now())
            kg_instance.relations.append(article_relation)
            kg_instance.relations_tracker[(entity_id,
                                           'mentioned_in', 
                                           article.permatitle, 
                                           'article')] = relation_id