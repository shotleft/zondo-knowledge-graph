from collections import Counter
from ..kg import kg_dataclasses as kg
from .. import hlp_functions as hlp
from ..cr import find_matching_entity_names

# These make up an 'ontology' loosely based on Wikidata.
rebel_flair_overview, PIds, QIds, property_details, item_details = hlp.get_wikidata_prepared_info('reference_info/wikidata_references.pkl')
           
            
def remove_self_relations(article: kg.Article):
    '''
    Remove relations where the subject and object are the same term.
    '''
    article.relations = [relation for relation in article.relations if relation.head_text != relation.tail_text]   
    
    
    
def cleanup_alternate_name_pairs(article: kg.Article):
    '''
    Takes in an Article instance, and examines the alternate_name relations, extracting only those
    that have a named entity on both sides of the relationship. The reason this function exists is
    to reduce the number of spurious alternate_name relations returned by Flair.
    '''
    relations_to_delete = []
    clean_entities = [{'text': entity.text, 'start_char': entity.start_char, 
                   'end_char': entity.end_char, 'ner_type': entity.ner_type} for entity in article.named_entities]
    # alternate_names = [relation for relation in article.relations if relation.relation_type == 'alternate_name']
    for relation in article.relations:
        if relation.relation_type == 'alternate_name':
            head = find_matching_entity_names([(relation.head_start_char, relation.head_end_char)], clean_entities, inner_match = True, return_tuples = True)
            tail = find_matching_entity_names([(relation.tail_start_char, relation.tail_end_char)], clean_entities, inner_match = True, return_tuples = True)
            # If both the head and the tail returned a named entity
            if all([head, tail]) and len(head) == 1 and len(tail) == 1:
                (head,) = head
                (tail,) = tail
                relation.head_text = head[0]
                relation.head_start_char = head[1][0]
                relation.head_end_char = head[1][1]
                relation.tail_text = tail[0]
                relation.tail_start_char = tail[1][0]
                relation.tail_end_char = tail[1][1]
            else:
                relations_to_delete.append(relation.relation_id)
    article.relations = [relation for relation in article.relations if relation.relation_id not in relations_to_delete]
    
    
def cleanup_duplicate_alternate_name_pairs(article: kg.Article):
    '''
    Takes in an Article instance, and examines the alternate_name relations, removing duplicate
    inverse alternate names, e.g. for the following only the first relation needs to be kept:
    National Prosecution Authority >> alternate_name >> NPA
    [884:914], [879:882].
    NPA >> alternate_name >> National Prosecution Authority
    [879:882], [884:914]
    '''
    alt_relations = [relation for relation in article.relations if relation.relation_type == 'alternate_name']
    inverse_duplicates = []
    for i, main in enumerate(alt_relations):
        for other in alt_relations[i+1:]:  # Start from the next relation after `main`
            if (main.head_start_char, main.head_end_char, main.tail_start_char, main.tail_end_char) == \
               (other.tail_start_char, other.tail_end_char, other.head_start_char, other.head_end_char):
                inverse_duplicates.append((main, other))
    duplicates_to_delete = []
    for duplicates in inverse_duplicates:
        for duplicate in duplicates:
            if len(duplicate.head_text) < len(duplicate.tail_text):
                duplicates_to_delete.append(duplicate.relation_id)
    article.relations = [relation for relation in article.relations if relation.relation_id not in duplicates_to_delete]
    
    
def populate_alt_names_mentions(article: kg.Article):
    '''
    For an article, get the alt_mentions (terms in the text like 'Zuma' and 'Jacob Zuma' which
    would need to be resolved to the main entity 'Jacob Zuma') as well as the alt_names which
    come from Flair's alternate_name relation and represent generally alternate names for entities
    like the 'National Prosecuting Authority' which is also referred to as the 'NPA'. Once these 
    have been populated we can remove the alternate_name from relations as we will not be 
    treating it as an actual relation.
    '''
    # DELETE 'strong' clause once we're past the sample dataset: it is to cater for a 
    # stray tag that crept in via Synopsis which was not cleaned upfront. This cleaning 
    # step will be built into the pre-processing going forward and the need for this 
    # tomfoolery will fall away.
    persons = [e for e in article.named_entities if e.ner_type == 'PERSON' and 'strong' not in e.text]
    overlap_dict = {}
    for main in persons:
        for comp in persons:
            # Check that the 2 entities are not identical
            # and check if they overlap
            if main.text != comp.text and main.text in comp.text:
                # If they do and the entity doesn't yet exist in our dict then add it
                if main.text not in overlap_dict:
                    overlap_dict[main.text] = {comp.text}
                # Otherwise add the additional reference found
                else:
                    overlap_dict[main.text].add(comp.text)
    # We only want to obtain short references with one overlapping long reference
    # as these are the only ones we can be sure about
    article.alt_mentions = {k: list(v)[0] for k, v in overlap_dict.items() if len(v) == 1}
    article.alt_names = {}
    # Old version
    # for relation in article.relations:
    #     if relation.relation_type == 'alternate_name':
    #         if not (relation.head_text.isupper() and relation.tail_text.isupper()) and relation.score > 0.999:
    #             article.alt_names[relation.tail_text] = relation.head_text
    
    # New version
    for relation in article.relations:
        if relation.relation_type == 'alternate_name':
            # Exclude instances where there is an acronym on both sides or the relation had a low score
            if not (relation.head_text.isupper() and relation.tail_text.isupper()) and relation.score > 0.999:
                # Put the shortest name on the left
                if (len(relation.tail_text) < len(relation.head_text)):
                    article.alt_names[relation.tail_text] = relation.head_text
                elif (len(relation.head_text) < len(relation.tail_text)):
                    article.alt_names[relation.head_text] = relation.tail_text
    
    article.relations = [relation for relation in article.relations if relation.relation_type != 'alternate_name']



def populate_clean_relation_texts(article: kg.Article):
    '''
    Populates 'clean' entities wherever possible, for example if 'Zuma' can be resolved to
    'Jacob Zuma' that is preferable. Similarly if 'NPA' can be resolved to 'National Prosecuting
    Authority' then we do so.
    '''
    alternates = article.alt_mentions | article.alt_names
    for relation in article.relations:
        relation.clean_head_text = alternates.get(relation.head_text, relation.head_text)
        relation.clean_tail_text = alternates.get(relation.tail_text, relation.tail_text)



def remove_ambiguous_relations(article: kg.Article):
    '''
    Optional function to remove relations where there is ambiguity about what the relation IS, for example:
    Justice >> officeholder >> Zondo
    Justice >> occupation of >> Zondo
    '''
    # First make a dict of unique head/tail keys and then for each key add the relation types found
    unique_relations_keys = {}
    for relation in article.relations:
        key = (relation.head_text, relation.tail_text)
        if unique_relations_keys.get(key, None) is None:
            unique_relations_keys[key] = [(relation.relation_id, relation.relation_type)]
        else:
            unique_relations_keys[key].append((relation.relation_id, relation.relation_type))
    # Example output
    # {('Peter Thabethe', 'Free State agriculture department'): [('xdbIRjpat', 'employer')],
    #   etc.
    #  ('Justice', 'Zondo'): [('HtG96z38U', 'officeholder'), ('AduKsTv6Q', 'occupation of')]}

    # Keys that have more than one relation should be deleted, get a list of them
    relations_to_delete = []
    for key, val in unique_relations_keys.items():
        if len(val) > 1:
            relations_to_delete.extend([item[0] for item in val])
    relations_to_delete
    # Example output
    # ['sTnXrWuWr', 'FSoAGmQwx', 'HtG96z38U', 'AduKsTv6Q']

    # Remove the identified relations
    article.relations = [relation for relation in article.relations if relation.relation_id not in relations_to_delete]

    
def find_overlapping_relations(relations_list : list) -> list:
    '''
    Helper function to find overlapping relation instances where both the head and tail overlap, for example:
        Raymond Zondo >> position held >> Deputy Chief Justice
        Raymond Zondo >> position held >> Chief Justice
    '''
    overlapping_relations = []
    n = len(relations_list)
    for i in range(n):
        for j in range(i + 1, n):
            relation1 = relations_list[i]
            relation2 = relations_list[j]
            if relation1.relation_type == relation2.relation_type:
                # Check for head text overlap
                head_overlap = hlp.tuple_overlap(
                    (relation1.head_start_char, relation1.head_end_char),
                    (relation2.head_start_char, relation2.head_end_char),
                    bidirectional=True
                )
                # Check for tail text overlap
                tail_overlap = hlp.tuple_overlap(
                    (relation1.tail_start_char, relation1.tail_end_char),
                    (relation2.tail_start_char, relation2.tail_end_char),
                    bidirectional=True
                )
                if head_overlap and tail_overlap:
                    overlapping_relations.append((relation1.relation_id, relation2.relation_id))
    return overlapping_relations
    
    

def cleanup_overlapping_relations(article: kg.Article) ->list[tuple]:
    '''
    If relations overlap, first try to get the matching named entity OR choose the longest relation available
    For example:
        Raymond Zondo >> position held >> Deputy Chief Justice
        Raymond Zondo >> position held >> Chief Justice
    '''

    overlapping_relations = find_overlapping_relations(article.relations)
    # Example output
    # [('AAS88jwrZ', 'jdD1tKMCy'), ('4khopEDNd', 'nEeOvzG5T')]

    relations_to_delete = []
    for overlap in overlapping_relations:
        comparisons = [relation for relation in article.relations if relation.relation_id in overlap]
        overlapping_heads = comparisons[0].head_text != comparisons[1].head_text
        overlapping_tails = comparisons[0].tail_text != comparisons[1].tail_text
        if overlapping_heads and not overlapping_tails:
            head_entity = hlp.find_entity_name(comparisons[0].head_start_char, comparisons[0].head_end_char, article.named_entities)
            relation_to_delete = comparisons[1].relation_id
            if head_entity is None:
                head_entity = hlp.find_entity_name(comparisons[1].head_start_char, comparisons[1].head_end_char, article.named_entities)
                relation_to_delete = comparisons[0].relation_id
                if head_entity is None:
                    if len(comparisons[0].head_text) > len(comparisons[1].head_text):
                        relation_to_delete = comparisons[1].relation_id
                    else:
                        relation_to_delete = comparisons[1].relation_id
            relations_to_delete.append(relation_to_delete)
        if overlapping_tails and not overlapping_heads:
            tail_entity = hlp.find_entity_name(comparisons[0].tail_start_char, comparisons[0].tail_end_char, article.named_entities)
            relation_to_delete = comparisons[1].relation_id
            if tail_entity is None:
                tail_entity = hlp.find_entity_name(comparisons[1].tail_start_char, comparisons[1].tail_end_char, article.named_entities)
                relation_to_delete = comparisons[0].relation_id
                if tail_entity is None:
                    if len(comparisons[0].tail_text) > len(comparisons[1].tail_text):
                        relation_to_delete = comparisons[1].relation_id
                    else:
                        relation_to_delete = comparisons[1].relation_id
            relations_to_delete.append(relation_to_delete)
        if overlapping_tails and overlapping_heads:
            tail_entity = hlp.find_entity_name(comparisons[0].tail_start_char, comparisons[0].tail_end_char, article.named_entities)
            if tail_entity is None:
                if len(comparisons[0].tail_text) > len(comparisons[1].tail_text):
                    relation_to_delete = comparisons[1].relation_id
                else:
                    relation_to_delete = comparisons[1].relation_id
                relations_to_delete.append(relation_to_delete)
            # Example output
            # ['jdD1tKMCy', 'nEeOvzG5T']

    # Remove the identified relations
    article.relations = [relation for relation in article.relations if relation.relation_id not in relations_to_delete]


def update_ner_mapping_items(labels_dict: dict) -> dict:
    '''
    Utility function to update subject or object NER mapping data to cater for
    multiple entity types or specified entity types in a comma-delimited list
    '''
    for key, value in labels_dict.items():
        if value == 'multiple':
            labels_dict[key] = []
        if len(value.split(',')) > 1:
            labels_dict[key] = value.split(',')
    return labels_dict
    
def make_head_labels() -> dict:
    '''
    Will generate a dict of relations and expected head labels (if any). Example output:
    {'position held': 'PERSON',
     'member of political party': 'PERSON',
     'country': [],
     'owner of': ['ORG', 'PERSON'],
      etc.}
    '''
    head_labels = hlp.make_lookup_dict_from_df(rebel_flair_overview, 'rebel description', 'subject NER mapping')
    head_labels = update_ner_mapping_items(head_labels)
    inverse_head_labels = hlp.make_lookup_dict_from_df(rebel_flair_overview, 'inverse description', 'object NER mapping')
    inverse_head_labels = update_ner_mapping_items(inverse_head_labels)
    head_labels.update(inverse_head_labels)
    return head_labels
    
def make_tail_labels() -> dict:
    '''
    Will generate a dict of relations and expected tail labels (if any). Example outputs:
    {'position held': 'POSITION',
     'member of political party': 'ORG',
     'country': 'GPE',
     'owner of': 'ORG',
      etc.}
    '''
    tail_labels = hlp.make_lookup_dict_from_df(rebel_flair_overview, 'rebel description', 'object NER mapping')
    tail_labels = update_ner_mapping_items(tail_labels)
    inverse_tail_labels = hlp.make_lookup_dict_from_df(rebel_flair_overview, 'inverse description', 'subject NER mapping')
    inverse_tail_labels = update_ner_mapping_items(inverse_tail_labels)
    tail_labels.update(inverse_tail_labels)
    return tail_labels


def most_frequent_type(type_list: list) -> str:
    '''
    If an entity like 'Guptas' occurs multiple times in the article's list of relations, it may do
    so with different head/tail types depending on the relation context. The assumption is that
    if it occurs more frequently as 'ORG' than it does as 'PERSON' then we'll treat it as an 'ORG'
    where the possibilities are open.
    Input would be all the variants for the head/tail types, e.g. ['ORG', 'ORG', ['ORG, 'PERSON'], []]
    and output would be the most frequently-used individual mention, e.g. 'ORG', if available.
    '''
    # Count the frequency of each type as long as it's not a list
    frequency = Counter([item if not isinstance(item, list) else None for item in type_list])
    
    # Get the type with the highest frequency if it's available
    if frequency:
        most_common = frequency.most_common(1)[0]  # Get the most common item and its count
        if most_common[1] > 1:
            return most_common[0]
    return None


def populate_node_types(article: kg.Article):
    headlabel_lookup = make_head_labels()
    taillabel_lookup = make_tail_labels()
    # Populate the head and tail types with initial Wikidata mappings (insofar as they are available)
    for relation in article.relations:
        relation.head_type = headlabel_lookup[relation.relation_type]
        relation.tail_type = taillabel_lookup[relation.relation_type]
        
        # First we will see if we can assess the correct node type based on how frequently a node was labelled with a type
        if isinstance(relation.head_type, list):
            entities_list = [r.head_type for r in article.relations if r.head_text == relation.head_text] + \
                            [r.tail_type for r in article.relations if r.tail_text == relation.head_text]
            entity_type = most_frequent_type(entities_list)
            if entity_type is not None:
                relation.head_type = entity_type
        if isinstance(relation.tail_type, list): 
            entities_list = [r.head_type for r in article.relations if r.head_text == relation.tail_text] + \
                            [r.tail_type for r in article.relations if r.tail_text == relation.tail_text]
            entity_type = most_frequent_type(entities_list)
            if entity_type is not None:
                relation.tail_type = entity_type
                
        # If we didn't manage to get a definitive node label based on frequency we can try getting it from named entities
        if isinstance(relation.head_type, list):
            entity_type = hlp.find_entity_type(relation.head_start_char, relation.head_end_char, article.named_entities)
            if entity_type is not None:
                relation.head_type = entity_type
        if isinstance(relation.tail_type, list):
            entity_type = hlp.find_entity_type(relation.tail_start_char, relation.tail_end_char, article.named_entities)
            if entity_type is not None:
                relation.tail_type = entity_type
    
    # And if there are still listy-type items left exclude them as ambiguous
    article.relations = [relation for relation in article.relations if not isinstance(relation.head_type, list) and not isinstance(relation.tail_type, list)]
    

FINAL_INCLUSIONS = list(rebel_flair_overview.loc[rebel_flair_overview['incl'] == 'Y', 'rebel description'])
FINAL_INCLUSIONS += ['alternate_name']     
    
def get_main_relations_only(article: kg.Article):
    '''
    Inverse relations are entailed in the main relations so we don't need to build
    all the reciprocals into the graph necessarily.
    '''
    article.relations = [relation for relation in article.relations if relation.relation_type in FINAL_INCLUSIONS]