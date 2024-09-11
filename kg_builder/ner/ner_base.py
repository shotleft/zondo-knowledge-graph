# ## Import required libraries
# ## -------------------------

import os
import re
import json
import numpy as np
import pandas as pd

import spacy
spacy.prefer_gpu()

import torch
import flair
from flair.nn import Classifier
from flair.data import Sentence

from ..kg import kg_dataclasses as kg
from .. import hlp_functions as hlp



# ## Define global variables
# ## -----------------------

# Models we'll be using for the experiments
MODEL_NAMES = {'spacy', 'flair'}

# Restrict outputs to entities of interest
ENTITIES = {'PERSON', 'GPE', 'LOC', 'EVENT', 'FAC', 'LAW', 'ORG'}

# To standardize NER outputs we want to exclude titles, articles and possessives
REPLACEMENTS = {'Prof': '', 'Prof.': '', 'Adv.': '', 'Dr.': '',
                'Mr': '', 'Ms.': '', ' SC': '', 'Miss': '',
                'Adv': '', 'The': '', 'Dr': '', 'Mr.': '',
                'Mrs': '', 'Mrs.': '', 'the': '', 'Rev': '',
                "'s": '', 'Rev.': '', 'Ms': '', 'Advocate': ''} # SC added as special legal title

# Compile the regex for REPLACEMENTS once so it runs faster
start_pattern = re.compile(r"^(?:" + "|".join(map(re.escape, REPLACEMENTS.keys())) + r")\s")
end_pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, REPLACEMENTS.keys())) + r")$")



# ## Define functions for performing NER
# ## -----------------------------------

def setup_ner_tagger(model_name: str):
    '''
        Sets up the named entity recognition tagger for the model specified.
        model_name must be a value contained in MODEL_NAMES: 'spacy', 'flair'.
        Returns the specified NER model, and model name.
    '''
    if model_name == 'spacy':
        ner_tagger = spacy.load('en_core_web_trf')
    elif model_name == 'flair':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ner_tagger = Classifier.load('flair/ner-english-ontonotes-large')
    else:
        raise ValueError(f'''Invalid model name. Please specify one of {MODEL_NAMES}''')
    return ner_tagger, model_name

def get_entities(articles: list[kg.Article], model_name: str, ner_tagger):
    '''
    Takes in a list of Article instances, and a model name and updates the Article 
    instances with NamedEntity instances. 
    '''
    if model_name == 'spacy':
        docs = [d for d in ner_tagger.pipe([article.article_text for article in articles], batch_size = 5)]
        for doc, article in zip(docs, articles):
            named_entities = [
                kg.NamedEntity(
                    entity_id=hlp.generate_uid(),
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    text=ent.text,
                    ner_type=ent.label_
                )
                for ent in doc.ents if ent.label_ in ENTITIES
            ]
            article.named_entities = named_entities
            
    elif model_name == 'flair':
        for article in articles:
            doc = Sentence(article.article_text)
            ner_tagger.predict(doc)
            labels = [tag for tag in doc.get_labels() if tag.value in ENTITIES]
            labels = [label.data_point.to_dict() for label in labels]
            named_entities = [
                kg.NamedEntity(
                    entity_id=hlp.generate_uid(),
                    start_char=ent['start_pos'],
                    end_char=ent['end_pos'],
                    text=ent['text'],
                    ner_type=[item['value'] for item in ent['labels']][0]
                )
                for ent in labels
            ]
            article.named_entities = named_entities
            
    else:
        raise ValueError(f'''Invalid model name. Please specify one of {MODEL_NAMES}''')



# ## Define functions for importing NER from Label Studio
# ## ----------------------------------------------------

def load_ner_from_label_studio(json_file: str, df: pd.DataFrame, annotations: bool) -> list[kg.Article]:
    '''
    Takes in a json file in Label Studio format, and a corresponding DataFrame which includes 
    article Id and AllText, as well as an indicator on whether to read annotations or 
    predictions, and returns a list of Article instances with NER annotations.
    - json_file : path to Label Studio json file
    - df: Dataframe containing Id and AllText columns 
    - annotations: If True read annotations (the HITL annotations), 
      else read predicitons which contains the model predictions
    Returns a list of Article instances containing NamedEntity instances from Label Studio.
    '''
    if not os.path.exists(json_file):
        raise ValueError(f'''Directory or file not found.''')
        
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    articles = []
    
    for item in data:
        article_text = item["data"]["text"]
        
        # Find the matching row in the DataFrame by comparing article_text
        matching_row = df[df['AllText'] == article_text]
        if not matching_row.empty:
            # Assuming 'Id' is unique and always present when there's a match
            article_id = matching_row['Id'].values[0]
        else:
            # For cases where no match is found
            continue
        
        named_entities = []
        if annotations:
            main_source = 'annotations'
        else:
            main_source = 'predictions'
        for annotation in item[main_source]:
            for result in annotation["result"]:
                entity_id = result["id"]
                start_char = result["value"]["start"]
                end_char = result["value"]["end"]
                text = result["value"]["text"]
                ner_type = result["value"]["labels"][0]  # Assuming only one label per entity
                
                named_entity = kg.NamedEntity(
                    entity_id=entity_id,
                    start_char=start_char,
                    end_char=end_char,
                    # Workaround required for bug where some text is not populated in the annotation file
                    text=article_text[start_char:end_char],
                    ner_type=ner_type
                )
                named_entities.append(named_entity)
        
        article = kg.Article(
            article_id=str(article_id),  # Ensure the ID is a string if it's not already
            article_text=article_text,
            named_entities=named_entities
        )
        
        articles.append(article)
    
    return articles



# ## Define functions for post-processing NER
# ## ----------------------------------------

def remove_extraneous(text: str) -> str:
    '''
    Takes in a piece of text and strips extraneous titles, articles and possessives as
    defined by REPLACEMENTS from the beginning and end of the text.
    '''
    text = start_pattern.sub(lambda m: REPLACEMENTS[m.group(0).strip()], text)
    text = end_pattern.sub(lambda m: REPLACEMENTS[m.group(0).strip()], text)
    return text.strip()


def flatten_list(listicle):
    '''
    Small helper function to return errors and non-errors in appropriate format.
    Used by strict_ner_metrics() and relaxed_ner_metrics().
    '''
    return [(item[0][0], item[0][1], item[0][2], item[1]) for item in listicle]


def ner_metrics(TP: list, FP: list, FN: list, article_prediction: kg.Article) -> dict:
    '''
    Takes in lists of true positives (TP), false positives (FP) and false negatives (FN) 
    and calculates precision (% of NEs found that are correct), recall (% of NEs in the
    article found) and F1  for an article. Use categories explained at:
    https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
    Returns:
        A dict of metric values including:
        - article id
        - precision numerator
        - precision denominator
        - recall numerator
        - recall denominator
        - precision
        - recall
        - f1 score
        - a list of errors
        - a list of non-errors'''
    # Compile metric categories - start with COR, MIS, SPU
    # and then adjust as per definitions
    COR = TP
    MIS = FN
    SPU = FP
    PAR = []
    to_pop_MIS = []
    to_pop_SPU = []
    for i, n in enumerate(MIS):
        for j, p in enumerate(SPU):
            # Check if there is overlap on mismatched tuples
            if hlp.tuple_overlap(n[0], p[0]):
                    # Otherwise add them to PAR and remove them from MIS and SPU
                    PAR.append(n)
                    to_pop_MIS.append(i)
                    to_pop_SPU.append(j)
    MIS = [n for i, n in enumerate(MIS) if i not in to_pop_MIS]
    SPU = [p for j, p in enumerate(SPU) if j not in to_pop_SPU]

    # Which then lets us calculate precision, recall and F1 according to the relaxed method
    p_numerator = len(COR) + 0.5 * len(PAR)
    p_denominator = len(COR + PAR + SPU)
    r_numerator = len(COR) + 0.5 * len(PAR)
    r_denominator = len(COR + PAR + MIS)
    precision = p_numerator / p_denominator
    recall = r_numerator / r_denominator
    f1 = 2 * (precision * recall) / (precision + recall)

    # And we save away the errors for review
    errors = [(pred, "MIS") for pred in MIS]
    errors.extend([(pred, "SPU") for pred in SPU])
    errors.extend([(pred, "PAR") for pred in PAR])

    # And for completeness also the non-errors
    non_errors = [(pred, "COR") for pred in COR]

    return {
        "article_id": article_prediction.article_id,
        "p_numerator": p_numerator,
        "p_denominator":p_denominator,
        "r_numerator": r_numerator,
        "r_denominator": r_denominator,
        "precision": precision, 
        "recall": recall,       
        "f1": f1,
        "errors": flatten_list(errors),
        "non-errors": flatten_list(non_errors)
    }

def calc_article_ner_metrics(article_prediction: kg.Article, article_annotations: list) -> dict:
    '''
    Takes in an Article instance (article_prediction) and a list of annotated Article instances
    (article_annotations). Finds the correct article in article_annotations to compare against, 
    and then returns precision, recall and F1 scores. 
    Returns:
        A dict of metric values including:
        - article id
        - precision numerator
        - precision denominator
        - recall numerator
        - recall denominator
        - precision
        - recall
        - f1 score
        - a list of errors
        - a list of non-errors
    '''
        
    # Find the right article to compare against in article_annotations
    article_annotations_map = {article.article_id: article for article in article_annotations}
    matched_article = article_annotations_map.get(article_prediction.article_id)

    # Get data points for comparison as sets
    ground_truth_entities = {((entity.start_char, entity.end_char), entity.ner_type, entity.text) for entity in matched_article.named_entities}
    predicted_entities = {((entity.start_char, entity.end_char), entity.ner_type, entity.text) for entity in article_prediction.named_entities}
        
    # The sets let us easily get TP, FP, FN
    TP = list(ground_truth_entities & predicted_entities)
    # These are ones that are in predicted_entities that are not found in the ground_truth_entities
    FP = list(predicted_entities - ground_truth_entities)
    # These are ones that are in ground_truth_entities but are not found in the predicted_entities
    FN = list(ground_truth_entities - predicted_entities)
    
    return ner_metrics(TP = TP, FP = FP, FN = FN, article_prediction = article_prediction)

def calc_corpus_ner_metrics(article_metrics: list[dict]) -> tuple[float, float, float]:
    '''
    Takes in a list of article metrics (output from calc_article_ner_metrics()) and returns
    overall precision, recall and f1 score for the corpus of articles.
    '''
    precision = np.mean([article['precision'] for article in article_metrics])
    recall = np.mean([article['recall'] for article in article_metrics])
    f1 = np.mean([article['f1'] for article in article_metrics])
    return (precision, recall, f1)