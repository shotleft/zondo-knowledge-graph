import math
import random
import string
import pickle
import pandas as pd
import pywikibot
from pywikibot import Site, PropertyPage, ItemPage



def load_json_data(filepath: str):
    '''
    Loads a json file as specified.
    '''
    with open(filepath, 'r') as file:
        return json.load(file)


def generate_uid():
    '''
    Generates a unique 9-digit id when required
    '''
    return ''.join(random.choices(string.ascii_letters + string.digits, k=9))


def tuple_overlap(tuple1: tuple, tuple2: tuple, bidirectional: bool = True) -> bool:
    '''
    If bidirectional: determine whether tuple ranges overlap in either direction.
        Example:
        tuple1 = (13, 19)
        tuple2 = (12, 19)
        >> evaluates to True
    If not bidirectional: determine whether the range of tuple2 is contained within the range of tuple1.
        Example:
        tuple1 = (13, 19)
        tuple2 = (12, 19)
        >> evaluates to False
    '''
    x1, x2 = tuple1
    y1, y2 = tuple2
    s1 = y1 >= x1 and y2 <= x2
    s2 = None
    if bidirectional:
        s2 = x1 >= y1 and x2 <= y2
    if s1 or s2:
        return True
    else:
        return False


def term_charspans(term, text):
    '''
    Get the character span start and end where a term first occurs in a piece of text.
    '''
    start = text.lower().find(term.lower())
    end = start + len(term)
    return [start, end]

    
def get_property_id(label: str) -> str:
    '''Get the property ID for the specified label'''
    site = Site("wikidata", "wikidata")
    search_results = site.search_entities(label, language='en', type='property')
    output_properties = []
    for result in search_results:
        output_properties.append((result['id'], result['label']))
    if len(output_properties) > 0:
        # Return the first matching property ID
        return [output_properties[0]]
    else:
        return None 
    
    
def get_inverse_property(property: str) -> dict:
    '''Get the inverse property (P1696) or inverse label (P7087) for a given property ID.'''
    site = Site('wikipedia:en')
    repo = site.data_repository()
    property_dict = {'main': property[0]}
    property_id = property[0][0]
    property_page = PropertyPage(repo, property_id)
    property_page.get()
    if 'P1696' in property_page.claims or 'P7087' in property_page.claims:
        inverse_properties = []
        inverse_property_claims = property_page.claims.get('P1696')
        if inverse_property_claims is None:
            inverse_property_claims = property_page.claims.get('P7087')
        for claim in inverse_property_claims:
            inverse_property = claim.getTarget()
            inverse_property_id = inverse_property.getID()
            inverse_property_label = inverse_property.labels['en'] if 'en' in inverse_property.labels else 'No label found'
            inverse_properties.append((inverse_property_id, inverse_property_label))
        if len(inverse_properties) > 0:
            property_dict['inverse'] = inverse_properties[0]
    return property_dict


def get_wd_relation_data(relations_list: list) -> list:
    '''
    Get Wikidata relation data including KBId and reciprocal label where applicable.
    '''
    relations_summary = []
    for relation in relations_list:
        property = get_property_id(relation)
        if property:
            relations_summary.append(get_inverse_property(property))
    return relations_summary


def get_property_details(PId: str) -> dict:
    '''
    Takes in a property ID (PId), e.g. P39, and returns a dictionary with the following keys:
    - description (str): description of the property
    - aliases (list): list of aliases of the property
    - subject_type_constraint (dict): which may indicate what node 
      label is appropriate for the subject of the relation
        - id (str): id of the subject type constraint 
        - label (str): label of the subject type constraint
    - value_type_constraint (dict): which may indicate what node 
      label is appropriate for the predicate of the relation
        - id (str): id of the value-type constraint 
        - label (str): label of the value-type constraint
    '''
    
    site = Site('wikidata', 'wikidata')
    repo = site.data_repository()
    property_page = PropertyPage(repo, PId)
    property_page.get()

    # First get the description and aliases
    description = property_page.descriptions.get('en', 'No description found')
    aliases = property_page.aliases.get('en', [])

    property_data = {
        'description': description,
        'aliases': aliases,
        'subject_type_constraint': None,
        'value_type_constraint': None
    }

    # Loop through property page claims
    for claim in property_page.claims.get('P2302', []):     # P2302 = 'property constraint'
        if 'P2308' in claim.qualifiers:                     # P2308 = 'class'
            # Check the type of constraint
            constraint_type = claim.getTarget().getID()
            class_item = claim.qualifiers['P2308'][0].getTarget()
            class_item.get() 
            class_label = class_item.labels.get('en', 'No label found')
            class_description = {
                'id': class_item.getID(),
                'label': class_label
            }
            # Get the constraints of interest
            if constraint_type == 'Q21503250':              # Q21503250 = subject type constraint
                property_data['subject_type_constraint'] = class_description
            # Value type constraint (Q21510865)
            elif constraint_type == 'Q21510865':            # Q21510865 = value-type constraint
                property_data['value_type_constraint'] = class_description

    return property_data


def get_item_details(QId: str) -> dict:
    '''
    Takes in an item ID (QId), e.g. Q125703439, and returns a dictionary with the following keys:
    - description (str): description of the property
    - aliases (list): list of aliases of the property
    '''
    site = Site('wikidata', 'wikidata')
    repo = site.data_repository()
    item_page = ItemPage(repo, QId)
    item_page.get()

    # Get the description and aliases
    description = item_page.descriptions.get('en', 'No description found')
    aliases = item_page.aliases.get('en', [])

    item_data = {
        'description': description,
        'aliases': aliases
    }

    return item_data



def chunk_long_articles(text, max_chunk_size = 20000):
    '''
    Takes in article text and returns overlapping chunk boundaries. Primarily
    required to prevent CUDA out of memory issues on very long articles.
    '''
    article_length = len(text)
    num_chunks = math.ceil(article_length / max_chunk_size)
    overlap = math.ceil((num_chunks * max_chunk_size - article_length) / 
                        max(num_chunks - 1, 1))
    
    chunk_boundaries = []
    start = 0
    for i in range(num_chunks):
        chunk_boundaries.append([start + max_chunk_size * i,
                                 start + max_chunk_size * (i + 1)])
        start -= overlap
    return chunk_boundaries


def make_lookup_dict_from_df(df: pd.DataFrame, key_col: str, value_col: str) -> dict:
    '''
    Takes in a df and 2 column names and returns a lookup dict 
    with key-value pairs
    '''
    df = df.copy()
    df = df[[key_col, value_col]].dropna()
    return dict(zip(df[key_col], df[value_col]))


def get_wikidata_prepared_info(path_to_pkl: str) -> [pd.DataFrame, list, list, dict, dict]:
    with open(path_to_pkl, 'rb') as file:
        rebel_flair_overview, PIds, QIds, property_details, item_details = pickle.load(file)
    return rebel_flair_overview, PIds, QIds, property_details, item_details


def find_entity_name(start_char: int, end_char: int, named_entities: list) -> str:
    '''
    Helper function to lookup the the exact entity name in named entities if there is a match.
    '''
    for entity in named_entities:
        if entity.start_char == start_char and entity.end_char == end_char:
            return entity.text
    return None


def find_entity_type(start_char: int, end_char: int, named_entities: list) -> str:
    '''
    Helper function to find an exact match on entity to get the type
    '''
    for entity in named_entities:
        if entity.start_char == start_char and entity.end_char == end_char:
            return entity.ner_type
    return None


def get_key_for_value(my_dict, value):
    '''
    Amongst others the alt_names dict is stored with the abbreviation as the key and the full name
    as the value, but sometimes we want to retrieve the abbreviation for the full name.
    '''
    for k, v in my_dict.items():
        if v.lower() == value.lower():
            return k
    return None

def sentence_in_range(sentence_indices, char):
    '''
    Takes in sentence_indices (a dict of sentence numbers and their start and end characters) 
    and a character index and returns the sentence number in which that character index occurs.
    '''
    for sentence, (start, end) in sentence_indices.items():
        if start <= char <= end:
            return sentence
    return None

def get_article_by_article_id(articles_list, article_id):
    '''
    Takes in a list of Article instances and an article_id and returns the matching Article instance.
    '''
    articles = [article for article in articles_list if article.article_id == article_id]
    if len(articles) > 0:
        return articles[0]
    return None


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
        start, stop = article.sentence_indices[sentence_in_range(article.sentence_indices, mention_start)]
        first_mention = article.article_text[start:stop]
    return first_mention

def track_relation_origin(relations, articles, relation_id):
    '''
    Use relation_id as a key to get details of relation found, and which (initial)
    sentence it was found in in the corpus.
    '''
    for relation in relations:
        if relation.RelationId == relation_id:
            first_instance = next(iter(relation.Instances))
            article_id = first_instance[0]
            sent_start, sent_end = first_instance[2]
            article = get_article_by_article_id(articles, article_id)
            sentence = article.article_text[sent_start: sent_end]
            print(f'''Relation {relation_id}: {relation.HeadName} >> {relation.Type} >> {relation.TailName}

Origin: {article.permatitle}
{sentence}''')