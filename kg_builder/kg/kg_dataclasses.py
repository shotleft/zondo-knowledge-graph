# ## Import required libraries
# ## -------------------------

from dataclasses import dataclass, asdict, field
from collections import defaultdict
import datetime
from typing import Optional, Union
import pandas as pd
import csv
import os
from google.cloud import storage
from .. import hlp_functions as hlp

# ## Define global variables
# ## -----------------------

EXCL_PUNCTUATION = '''!"#$%&\'()*+,./:;<=>?@[\\]^`{|}~'''
CLIENT = storage.Client(project='cas-lake')
BUCKET = CLIENT.get_bucket('zondo-kg')


# ## Define data classes required for initial article processing using base models
# ## -----------------------------------------------------------------------------

@dataclass
class NamedEntity():
    '''
    This class holds information on a single named entity within an article.
    '''
    entity_id: str
    start_char: int
    end_char: int
    text: str
    ner_type: str
    
    def to_dict(self) -> dict[str, any]:
        '''
        Returns a NamedEntity instance in dict format.
        '''
        return asdict(self)


@dataclass
class Mention():
    '''
    This class holds information on a single mention within a coreference cluster from an article.
    '''
    mention_id: str
    start_char: int
    end_char: int
    text: str
    
    def to_dict(self) -> dict[str, any]:
        '''
        Returns a Mention instance in dict format.
        '''
        return asdict(self)


@dataclass
class CRCluster():
    '''
    This class holds information on a single coreference cluster within an article.
    '''
    cluster_id: str
    resolved_text: str = None
    mentions: list[Mention] = None

    def to_dict(self) -> dict[str, any]:
        '''
        Returns a CRCluster instance in dict format.
        '''
        return asdict(self)


@dataclass
class Relation():
    '''
    This class holds information on a single relation within an article.
    '''
    # These data points are gathered first
    relation_id: str
    head_start_char: int
    head_end_char: int
    head_text: str
    tail_start_char: int
    tail_end_char: int
    tail_text: str
    relation_type: str
    direction: str
    # WTF???
    head_id: Optional[str] = None
    tail_id: Optional[str] = None
    # Afterwards we derive the type
    head_type: Optional[str] = None
    tail_type: Optional[str] = None
    # And optionallly store a score for Flair alternate_name relations
    score: Optional[float] = None
    # And finally we want to store clean text if possible, e.g.
    # where Zuma is resolved to Jacob Zuma in the relation so
    # that some level of disambiguation is achieved in the mentions
    clean_head_text: Optional[str] = None
    clean_tail_text: Optional[str] = None
    
    def to_dict(self) -> dict[str, any]:
        '''
        Returns a Relation instance in dict format.
        '''
        return asdict(self)


@dataclass
class Article():
    '''
    This class holds information on a single article, including source data points, 
    information required for interim processing like parse_info and sentence_indeices,
    as well as potential named entities, coreference clusters, and relations.
    '''
    article_id: str
    article_text: str
    permatitle: str = None
    parse_info: list = None
    sentence_indices: dict = None
    named_entities: list[NamedEntity] = None
    cr_clusters: list[CRCluster] = None
    relations: list[Relation] = None
    alt_names: dict = None
    alt_mentions: dict = None
    
    def to_dict(self) -> dict[str, any]:
        '''
        Returns an Article instance in dict format.
        '''
        return asdict(self)
    
    def named_entities_to_list(self) -> list[dict]:
        '''
        Returns named entities in a list format.
        '''
        return [{'text': entity.text, 
                 'start_char': entity.start_char, 
                 'end_char': entity.end_char, 
                 'ner_type': entity.ner_type} for entity in self.named_entities]
    
    def ner_labelstudio(self) -> dict[str, any]:
        '''
        Returns a dictionary of named entities for export to JSON.
        '''
        if self.named_entities is not None:
            return {
                "id": self.article_id,
                "data": {
                    "text": self.article_text
                },
                "predictions": [
                    {
                        "result": [
                            {
                                "value": {
                                    "start": entity.start_char,
                                    "end": entity.end_char,
                                    "text": entity.text,
                                    "labels": [entity.ner_type]
                                },
                                "id": entity.entity_id,
                                "from_name": "label",
                                "to_name": "text",
                                "type": "labels",
                                "origin": "manual"
                            } for entity in self.named_entities
                        ]
                    }
                ]
            }
    
    def cr_labelstudio(self) -> dict[str, any]:
        '''
        Returns a dictionary of coreference clusters for export to JSON.
        '''
        if self.cr_clusters is not None:
            result = []
            cluster_num = 1
            for cluster in self.cr_clusters:
                cluster_labels = f'''Cluster {cluster_num}'''
                for mention in cluster.mentions:
                    result.append(
                        {
                            "value": {
                                "start": mention.start_char,
                                "end": mention.end_char,
                                "text": mention.text,
                                "labels": [cluster_labels]
                            },
                            "meta": {
                                "text": "default"
                            },
                            "id": mention.mention_id,
                            "from_name": "label",
                            "to_name": "text",
                            "type": "labels",
                            "origin": "manual"
                        }
                    )
                cluster_num += 1
            return {
                "id": self.article_id,
                "data": {
                    "text": self.article_text
                },
                "predictions": [
                    {
                        "result": result
                    }
                ]
            }
        
    def re_labelstudio(self) -> dict[str, any]:
        '''
        Returns a dictionary of relations for export to JSON.
        '''
        if self.relations is not None:
            predictions = []
            span_to_id = {}
            result = []
            id_counter = 1  # Initialize a counter for IDs

            for item in self.relations:
                head_span = (item.head_start_char, item.head_end_char)
                tail_span = (item.tail_start_char, item.tail_end_char)

                # Check if head entity has not been added before
                if head_span not in span_to_id:
                    span_to_id[head_span] = {"id": str(id_counter)}
                    id_counter += 1  # Increment the counter
                    item.head_id = span_to_id[head_span]['id']

                    head_result = {
                        "value": {
                            "start": item.head_start_char,
                            "end": item.head_end_char,
                            "text": item.head_text,
                            "labels": ["entity"]
                        },
                        "id": span_to_id[head_span]["id"],
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "origin": "manual"
                    }
                    result.append(head_result)
                else:
                    # If already added, just update the head_id
                    item.head_id = span_to_id[head_span]['id']

                # Check if tail entity has not been added before
                if tail_span not in span_to_id:
                    span_to_id[tail_span] = {"id": str(id_counter)}
                    id_counter += 1  # Increment the counter
                    item.tail_id = span_to_id[tail_span]['id']

                    tail_result = {
                        "value": {
                            "start": item.tail_start_char,
                            "end": item.tail_end_char,
                            "text": item.tail_text,
                            "labels": ["entity"]
                        },
                        "id": span_to_id[tail_span]["id"],
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "origin": "manual"
                    }
                    result.append(tail_result)
                else:
                    # If already added, just update the tail_id
                    item.tail_id = span_to_id[tail_span]['id']

            # Add relations
            for item in self.relations:
                relation_result = {
                    "from_id": item.head_id,
                    "to_id": item.tail_id,
                    "type": "relation",
                    "direction": item.direction,
                    "labels": [item.relation_type]
                }
                result.append(relation_result)

            predictions.append({
                "id": str(id_counter),  # Use the counter for annotation ID as well
                "result": result
            })

            # Return a dictionary directly for each article
            return {
                "id": self.article_id,  # Assuming article_id is already suitable
                "predictions": predictions,
                "data": {
                    "text": self.article_text
                }
            }

    
    def to_labelstudio(self, tasktype: str) -> dict[str, any]:
        '''
        This function converts an Article instance and the specified tasktype to the format 
        required by Label Studio. Variable tasktype must be one of 'named_entities', 'cr_clusters', 
        or 'relations'. Returns a dict ready for export to JSON.
        '''
        
        valid_tasktypes = ['named_entities', 'cr_clusters', 'relations']
        if tasktype not in valid_tasktypes:
            raise ValueError(f'''Invalid tasktype. Please specify one of {valid_tasktypes}.''')
        
        if tasktype == 'named_entities':
            return self.ner_labelstudio()
        
        if tasktype == 'cr_clusters':
            return self.cr_labelstudio()
        
        if tasktype == 'relations':
            return self.re_labelstudio()
        
    
    def to_cr_evalformat(self) -> list:
        '''
        Convert CRCluster instances in an article to the specified list of list of lists format
        which the corefeval library requires for evaluation.
        
        NOTE: the order of cluster lists does not affect the score outcome, i.e. the following two
        examples produce the same final scores:
        
        Example 1:
        gold = [[[50, 50], [27, 27], [29, 29]], [[0, 1], [7, 13]]]
        pred = [[[50, 50], [27, 27], [29, 29]], [[0, 1], [42, 42], [7, 13]], 
               [[200, 201], [242, 242], [72, 132]]]
               
        Example 2:
        gold = [[[50, 50], [27, 27], [29, 29]], [[0, 1], [7, 13]]]
        pred = [[[50, 50], [27, 27], [29, 29]], [[200, 201], 
               [242, 242], [72, 132]], [[0, 1], [42, 42], [7, 13]]]
        '''
        
        cluster_spans = []
        for cluster in self.cr_clusters:
            mention_spans = [[mention.start_char, mention.end_char] for mention in cluster.mentions]
            cluster_spans.append(mention_spans)
        return cluster_spans   
    
    
    def print_named_entities(self, entity_types: list = None):
        '''
        Print entities for an article to screen - either all entity types or just those specified.
        '''
        if entity_types is not None:
            ner_set = set([entity.ner_type for entity in self.named_entities])
            if len(set(entity_types) & ner_set) > 0:
                print(f'''
Article {self.article_id}
{self.article_text[0:44]}
--------------------------------------------''')
                for entity in self.named_entities:
                    if entity.ner_type in entity_types:
                        print(f'''{entity.text}, {entity.ner_type}, [{entity.start_char}:{entity.end_char}]''')
        else:
            print(f'''
Article {self.article_id}
{self.article_text[0:44]}
--------------------------------------------''')
            for entity in self.named_entities:
                print(f'''{entity.text}, {entity.ner_type}, [{entity.start_char}:{entity.end_char}]''')
    
    
    def print_relations(self, relation_types: list = None):
        '''
        Print relations for an article to screen - either all relation types or just those specified.
        '''
        if relation_types is not None:
            relationset = set([relation.relation_type for relation in self.relations])
            if len(set(relation_types) & relationset) > 0 or relation_types is None:
                print(f'''
Article {self.article_id}
{self.article_text[0:44]}
--------------------------------------------''')
                for relation in self.relations:
                    if relation.relation_type in relation_types:
                        print(f'''{relation.head_text} {'>>'if relation.direction == 'right' else '<<'} {relation.relation_type} {'>>'if relation.direction == 'right' else '<<'} {relation.tail_text}
[{relation.head_start_char}:{relation.head_end_char}], [{relation.tail_start_char}:{relation.tail_end_char}]''')
        else:
            print(f'''
Article {self.article_id}
{self.article_text[0:44]}
--------------------------------------------''')
            for relation in self.relations:
                print(f'''{relation.head_text} {'>>'if relation.direction == 'right' else '<<'} {relation.relation_type} {'>>'if relation.direction == 'right' else '<<'} {relation.tail_text}
[{relation.head_start_char}:{relation.head_end_char}], [{relation.tail_start_char}:{relation.tail_end_char}]                ''')

                
    def print_cr_clusters(self, post_processing: bool = True):
        '''
        Print coreference clusters to screen.
        '''
        print(f'''
Article {self.article_id}
{self.article_text[0:44]}
--------------------------------------------''')
        
        if post_processing:
            
            for cluster in self.cr_clusters:
                if 'AAAmbiguous' in cluster.resolved_text:
                    print(f'''
Ambiguous cluster
-------------------''')
                else:
                    print(f'''
Defined cluster
-------------------''')
                print({text for text in cluster.resolved_text if text != 'AAAmbiguous'})
                for mention in cluster.mentions[0:5]:
                    print(f'''{mention.text}, [{mention.start_char}:{mention.end_char}]''')
        else:
            
            for i, cluster in enumerate(self.cr_clusters):
                print(f'''\nCluster {i}: ''')
                for mention in cluster.mentions[0:5]:
                    print(f'''{mention.text}, [{mention.start_char}:{mention.end_char}]''')
            

# ## Define function for creating a list of new Article instances 
# ## ------------------------------------------------------------

def make_articles(df: pd.DataFrame) -> list[Article]:
    '''
    Takes in a DataFrame containing article Id and AllText
    '''
    articles = []
    for row in df.itertuples():
        articles.append(Article(article_id=row.Id, article_text = row.AllText, permatitle = row.Permatitle))
    return articles


# ## Define classes required for final KG contruction and processing
# ## ---------------------------------------------------------------

@dataclass
class KGEntity():
    '''
    This class holds information on a single named entity in the KG.
    '''
    EntityId: str
    Name: str
    Type: str
    Updated: datetime
    WDId: str = None
    WDUrl: str = None
    WDDescription: str = None
    WDRetry: int = 0
    WDSource: str = None
    WDRetryArticles: set = field(default_factory=set)
    PywikiStatus: Union[str, bool] = None
    AlsoKnownAs: set = field(default_factory=set)
    
    
@dataclass
class KGRelation():
    '''
    This class holds information on a single relation in the KG.
    '''
    RelationId: str
    HeadId: str
    HeadName: str
    Type: str
    TailId: str
    TailName: str
    Updated: datetime
    Instances: dict # (ArticleId, Permatitle) : Occurences
    WDType: str = None
    

class ChangeLogDict(dict):
    '''
    For the KGData entities_tracker we need a dict that can track changes made, so
    that they can be applied downstream in the relevant relations as well.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.changes = []

    def __setitem__(self, key, value):
        current_time = datetime.datetime.now()
        previous = self.get(key, None)
        super().__setitem__(key, value)
        if previous is not None and previous != value:
            self.changes.append({
                'key': key,
                'previous': previous,
                'new_value': value,
                'datetime': current_time,
                'type': 'update'
            })
    
@dataclass
class KGData:
    '''
    This class holds information on the full and final KG.
    '''
    entities: list[KGEntity] = field(default_factory=list)
    entities_tracker: ChangeLogDict = field(default_factory=ChangeLogDict)
    relations: list[KGRelation] = field(default_factory=list)
    relations_tracker: dict = field(default_factory=dict)
    wd_tracker: dict = field(default_factory=dict)
    
    def get_entity_by_id(self, entity_id):
        for i, entity in enumerate(self.entities):
            if entity.EntityId == entity_id:
                return i, entity
    
    def get_relation_by_id(self, relation_id):
        for i, relation in enumerate(self.relations):
            if relation.RelationId == relation_id:
                return i, relation
            
            
    def prepare_kg_nx_files(self, run_start, folder, file_description):
        '''
        Prepares entities and relations for export in a format suited to networkx.
        '''
        filename = f'''{folder}/{file_description}_entities.csv'''
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['EntityId', 
                             'Name', 
                             'Type',
                             'AlsoKnownAs', 
                             'WDId', 
                             'WDUrl', 
                             'WDDescription', 
                             'WDSource'])  # Writing headers
            for entity in self.entities:
                if entity.Type != 'ARTICLE':
                    writer.writerow([entity.EntityId, 
                                     entity.Name,
                                     entity.Type,
                                     entity.AlsoKnownAs if len(entity.AlsoKnownAs) else None,
                                     entity.WDId,
                                     entity.WDUrl,
                                     entity.WDDescription,
                                     entity.WDSource])

        print('nx CSV files exported for entities.')

        filename = f'''{folder}/{file_description}_relations.csv'''

        mapping_dict = {}
        for entity in self.entities:
            mapping_dict[entity.EntityId] = {
                'Name': entity.Name,
                'Type': entity.Type
            }

        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['RelationId',
                             'HeadId',
                             'TailId',
                             'Weight',
                             'HeadName',
                             'Type',
                             'TailName'
                             ])  # Writing headers
            for relation in self.relations:
                if relation.Type != 'mentioned_in':
                    writer.writerow([relation.RelationId, 
                                     relation.HeadId,
                                     relation.TailId,    
                                     len(relation.Instances.keys()),
                                     mapping_dict[relation.HeadId]['Name'],
                                     relation.Type,
                                     mapping_dict[relation.TailId]['Name'],
                                     ])

        print('nx CSV files exported for relations.')
        
        
    def prepare_kg_neo4j_files(self, run_start, folder):
        '''
        Prepares entities and relations for export in a format suited to neo4j. 
        Exports to local folder (folder) as well as pre-defined GCP bucket. 
        '''
        excl_punctuation = str.maketrans('', '', EXCL_PUNCTUATION)
        # Grouping entities by Type
        entity_groups = defaultdict(list)
        for entity in self.entities:
            if entity.Updated >= run_start:
                entity_groups[entity.Type].append(entity)

        # Writing to CSV files
        for entity_type, items in entity_groups.items():
            filename = f'''{folder}/{entity_type.translate(excl_punctuation)}.csv'''
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['EntityId', 
                                 'Name', 
                                 'AlsoKnownAs', 
                                 'WDId', 
                                 'WDUrl', 
                                 'WDDescription', 
                                 'WDSource'])  # Writing headers
                for item in items:
                    writer.writerow([item.EntityId, 
                                     item.Name,
                                     list(item.AlsoKnownAs) if len(item.AlsoKnownAs) else None,
                                     item.WDId,
                                     item.WDUrl,
                                     item.WDDescription,
                                     item.WDSource])

        print('neo4j CSV files exported for entities.')
        print(f'''neo4j CSV file names: {entity_groups.keys()}''')

        # Grouping relations by Type
        relation_groups = defaultdict(list)
        for relation in self.relations:
            if relation.Updated >= run_start:
                relation_groups[relation.Type].append(relation)

        # Writing to CSV files
        relation_files = []
        for relation_type, items in relation_groups.items():
            base_name = relation_type.translate(excl_punctuation).replace(' ', '_')
            filename = f'''{folder}/{base_name}.csv'''
            relation_files.append(base_name)
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['RelationId', 'HeadId', 'TailId', 'Weight'])  # Writing headers
                for item in items:
                    writer.writerow([item.RelationId, item.HeadId, item.TailId, sum(item.Instances.values())])

        print('neo4j CSV files exported for relations.')
        print(f'''neo4j CSV file names: {relation_files}''')

        files_for_bucket = [file for file in os.listdir(folder) if file.endswith('csv')]

        for file in files_for_bucket:
            blob = BUCKET.blob(file)
            blob.upload_from_filename(f'''{folder}/{file}''')

        print('neo4j CSV files copied to GCP bucket.')