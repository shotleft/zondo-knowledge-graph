# ## Import required libraries
# ## -------------------------

import os
import re
import json
import random
import string
import numpy as np
import pandas as pd

import spacy
spacy.prefer_gpu()
from fastcoref import spacy_component

from corefeval import get_metrics

from ..kg import kg_dataclasses as kg
from .. import hlp_functions as hlp


# ## Define functions for performing CR
# ## ----------------------------------

def setup_cr_tagger(model_name: str):
    '''
        Sets up the coreference resolution tagger for the model specified.
        model_name must be one of 'fastcoref', 'lingmess'.
        Returns the specified CR model, and model name.
    '''
    cr_tagger = spacy.load('en_core_web_trf')
    if model_name == 'fastcoref':
        cr_tagger.add_pipe('fastcoref')
    elif model_name == 'lingmess':
        cr_tagger.add_pipe(
           "fastcoref", 
           config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref'}
        )
    else:
        raise ValueError(f'''Invalid model name. Please specify one of {MODEL_NAMES}''')
    return cr_tagger, model_name

def get_clusters(articles: list[kg.Article], model_name: str, cr_tagger):
    '''
    Takes in a list of Article instances, and a model_name and updates the Article instances with 
    CRCluster instances included.
    '''
    docs = [d for d in cr_tagger.pipe([article.article_text for article in articles], batch_size = 5)]
    for doc, article in zip(docs, articles):
        raw_clusters = doc._.coref_clusters
        cr_clusters = []
        for cluster in raw_clusters:
            mentions = [kg.Mention(
                                mention_id=hlp.generate_uid(),
                                start_char=mention[0],
                                end_char=mention[1],
                                text=article.article_text[mention[0]:mention[1]],
                    )
                    for mention in cluster
                ]
            cr_clusters.append(kg.CRCluster(cluster_id = hlp.generate_uid(), mentions = mentions))
        article.cr_clusters = cr_clusters



# ## Define functions for importing CR from Label Studio
# ## ---------------------------------------------------

def load_cr_from_label_studio(json_file: str, df: pd.DataFrame, annotations: bool) -> list[kg.Article]:
    '''
    Takes in a json file in Label Studio format, and a corresponding DataFrame which includes 
    article Id and AllText, as well as an indicator on whether to read annotations or 
    predictions, and returns a list of Article instances with CR annotations.
    - json_file : path to Label Studio json file
    - df: Dataframe containing Id and AllText columns 
    - annotations: If True read annotations (the HITL annotations), 
      else read predicitons which contains the model predictions
    Returns a list of Article instances containing CRCluster instances from Label Studio.
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
            article_id = matching_row['Id'].values[0]
        else:
            continue  # Skip this article if no match is found
        
        cr_clusters = []
        cluster_dict = {}
        main_source = 'annotations' if annotations else 'predictions'
        
        for annotation in item[main_source]:
            for result in annotation["result"]:
                cluster_label = result["value"]["labels"][0]  # Assuming each mention belongs to a single cluster
                if cluster_label not in cluster_dict:
                    cluster_dict[cluster_label] = []
                cluster_dict[cluster_label].append(kg.Mention(
                    mention_id=result["id"],
                    start_char=result["value"]["start"],
                    end_char=result["value"]["end"],
                    text=result["value"]["text"]
                ))
        
        for cluster_id, mentions in cluster_dict.items():
            cr_clusters.append(kg.CRCluster(cluster_id=cluster_id, mentions=mentions))
        
        articles.append(kg.Article(article_id=article_id, article_text=article_text, cr_clusters=cr_clusters))
    
    return articles


def calc_article_cr_metrics(article_prediction: kg.Article, article_annotations: list) -> dict:
    '''
    Takes in an Article instance (article_prediction) and a list of annotated Article instances
    (article_annotations). Finds the correct article in article_annotations to compare against. 
    Uses the CoNLL standard (average of MUC, B3 and CEAF-e F1's) to calculate F1 as
    implemented by the corefeval library.
    Returns:
        A dict of metric values including:
        - article id
        - avg_f1
    '''
    
    # Find the right article to compare against in article_annotations
    article_annotations_map = {article.article_id: article for article in article_annotations}
    matched_article = article_annotations_map.get(article_prediction.article_id)
    
    pred = article_prediction.to_cr_evalformat()
    gold = matched_article.to_cr_evalformat()
    
    avg_f1 = get_metrics(pred, gold, verbose = False)
    
    return {
        "article_id": article_prediction.article_id,
        "avg_f1": avg_f1[0]
    }


def calc_corpus_cr_metrics(article_metrics: list[dict]) -> float:
    '''
    Takes in a list of article metrics (output from calc_article_cr_metrics()) and returns
    overall average f1 score for the corpus of articles.
    '''
    corpus_avg_f1 = np.mean([article['avg_f1'] for article in article_metrics])
    return corpus_avg_f1
