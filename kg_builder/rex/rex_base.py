# ## Import required libraries
# ## -------------------------

import os
import json
import numpy as np
import pandas as pd

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import math
import torch

from flair.splitter import SegtokSentenceSplitter
from flair.nn import Classifier

from ..kg import kg_dataclasses as kg
from .. import hlp_functions as hlp



# ## Define global variables
# ## -----------------------

# Models we'll be using for the experiments
MODEL_NAMES = {'rebel', 'flair'}


# These items have been prepared in advance through a review and annotation process and use of PyWikibot
# to obtain additional information on relation Ids, inverses and additional property and item details
# (see hlp_functions.py for further details). These make up an 'ontology' loosely based on Wikidata.
# rebel_flair_overview                Prepared DataFrame containing Rebel & Flair relations
# PIds                                Property Ids (with P* prefix) from Wikidata (where available)
# QIds                                Item Ids (with Q* prefix) from Wikidata (where available)
# property_details                    Additional information on Property Ids from Wikidata (where available) including:
#                                       - description
#                                       - aliases
#                                       - subject_type_constraint - indicates subject node label type
#                                       - value_type_constraint - indicates object node label type
# item_details                        Additional information on Property Ids from Wikidata (where available) including:
#                                       - description
#                                       - aliases
rebel_flair_overview, PIds, QIds, property_details, item_details = hlp.get_wikidata_prepared_info('reference_info/wikidata_references.pkl')


# ## Define functions for performing RE
# ## ----------------------------------

def setup_rex_tagger(model_name: str):
    '''
        Sets up the relation extractor for the model_name specified.
        Must be a value contained in MODEL_NAMES.

        Returns:
            For rebel:
                The specified RE model, model tokenizer, device, and model name.
            For flair:
                The specified RE model, NE model, splitter, device, and model name.
    '''
    if model_name == 'rebel':
        # Check if GPU is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load the model and tokenizer (https://huggingface.co/Babelscape/rebel-large)
        rex_tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
        rex_tagger = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
        # Move model to required device
        rex_tagger.to(device)
        return rex_tagger, rex_tokenizer, device, model_name
    elif model_name == 'flair':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        splitter = SegtokSentenceSplitter()
        ner_tagger = Classifier.load('flair/ner-english-ontonotes-large')
        rex_tagger = Classifier.load('relations')
        # Move model to required device
        ner_tagger.to(device)
        rex_tagger.to(device)
        return rex_tagger, ner_tagger, splitter, device, model_name
    else:
        raise ValueError(f'''Invalid model name. Please specify one of {MODEL_NAMES}''')



def rebel_get_triples(text):
    '''
    Function to extract relations from the rebel model outputs as
    supplied at https://huggingface.co/Babelscape/rebel-large
    '''
    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations

def rebel_get_relations(article: kg.Article, rex_tokenizer, rex_tagger, device, chunk, span_length: int =128) -> kg.Article:
    '''
        Extracts the relations from a piece of text, i.e. an article. Based on code supplied at
        https://medium.com/nlplanet/building-a-knowledge-base-from-texts-a-full-practical-example-8dbbffb912fa
        but enhanced to standardize outputs on character spans rather than tokens
        
        Arguments:
            article: Article instance containing article_text
            rex_tokenizer: performs pre-processing tokenization
            rex_tagger: performs relation extraction
            device (str): 'cuda' if GPU is available, else 'cpu'
            span_length (int): Relations are extracted from overlapping spans of
                text. This can be considered a hyperparameter of sorts, in that
                changing the span length affects the model performance. Measured 
                in number of tokens
    '''
    # Create an empty dict to hold unique relations (because we are examining 3 outputs
    # for every span (num_return_sequences) duplicates may occur
    article_text = article.article_text[chunk[0]:chunk[1]]
    if not article.relations:
        unique_relations_dict = {}
    else:
        unique_relations_dict = {
            (relation.head_start_char, relation.head_end_char, 
             relation.head_text, relation.tail_start_char, 
             relation.tail_end_char, relation.tail_text,
             relation.relation_type, relation.direction): relation
            for relation in article.relations
        }
    
    # First tokenize the text
    initial_inputs = rex_tokenizer([article_text], return_tensors="pt")
    
    # Then get the number of tokens for the text
    # Example output 364 ≈ 364 words
    num_tokens = len(initial_inputs["input_ids"][0])
    
    # Then get the number of spans  for the text, given the span length
    # Example output 3
    num_spans = math.ceil(num_tokens / span_length)
    
    # Then get the number of tokens to overlap spans by
    # Example output 10
    overlap = math.ceil((num_spans * span_length - num_tokens) / 
                        max(num_spans - 1, 1))
    
    # And finally get the span boundaries - the start and end tokens 
    # for each span that the model will evaluate
    # Example output
    #    [[0, 128], [118, 246], [236, 364]])
    spans_boundaries = []
    start = 0
    for i in range(num_spans):
        spans_boundaries.append([start + span_length * i,
                                 start + span_length * (i + 1)])
        start -= overlap
        
    # Rebel's outputs are expressed as token spans, however the rest of our data
    # uses character spans (also required by Label Studio) so we need to be able
    # to convert the outputs to character spans.
    # A method token_to_chars() is provided, however the first and last charspans 
    # are None which can cause problems with the retrieval of start and end spans.
    # It SEEMS(!) to work to just fill the first and last Nones with the first and
    # last available inputs
    
    # Example output:
    #    [None,
    #     CharSpan(start=0, end=5),
    #     CharSpan(start=6, end=10),
    #     CharSpan(start=11, end=12), etc.
    #     CharSpan(start=1678, end=1679),
    #     CharSpan(start=1680, end=1680),
    #     None]
    charspans = [initial_inputs.token_to_chars(i) for i in range(0, num_tokens)]
    # Example output:
    #    [CharSpan(start=0, end=5),
    #     CharSpan(start=0, end=5),
    #     CharSpan(start=6, end=10),
    #     CharSpan(start=11, end=12), etc.
    #     CharSpan(start=1678, end=1679),
    #     CharSpan(start=1680, end=1680),
    #     None]    
    charspans[0] = charspans[1]
    # Example output:
    #    [CharSpan(start=0, end=5),
    #     CharSpan(start=0, end=5),
    #     CharSpan(start=6, end=10),
    #     CharSpan(start=11, end=12), etc.
    #     CharSpan(start=1678, end=1679),
    #     CharSpan(start=1680, end=1680),
    #     CharSpan(start=1680, end=1680)] 
    charspans[-1] = charspans[-2]

    # Transform the initial_inputs into inputs per span
    tensor_ids = [initial_inputs["input_ids"][0][boundary[0]:boundary[1]]
                  for boundary in spans_boundaries]
    tensor_masks = [initial_inputs["attention_mask"][0][boundary[0]:boundary[1]]
                    for boundary in spans_boundaries]
    
    # Reformat the tensors and move to GPU if available
    inputs = {
        "input_ids": torch.stack(tensor_ids).to(device),
        "attention_mask": torch.stack(tensor_masks).to(device)
    }

    # Generate the relations as per documentation - 3 independently computed
    # return sequences (num_return_sequences) are returned for each element in the 
    # batch as per https://huggingface.co/docs/transformers/en/main_classes/text_generation
    num_return_sequences = 3
    gen_kwargs = {
        "max_length": 128,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": num_return_sequences
    }
    generated_tokens = rex_tagger.generate(
        **inputs,
        **gen_kwargs,
    )

    # Decode the relations as per documentation The length of decoded_preds
    # will be num_spans * num_return_sequences, in other words for the first
    # span 3 predictions are returned, for the second span 3 predictions
    # are returned, and so on.
    decoded_preds = rex_tokenizer.batch_decode(generated_tokens,
                                           skip_special_tokens=False)

    # Assemble the final relation data
    
    # spans_boundaries was needed by the model as-is above. But thereafter we want to use it
    # to get the charspans relevant to a sentence. It is one element too long which results in list
    # index out of range errors so here we make the last span boundary the length of charspans - 1.
    # Such a dark hack but nonetheless it seems to work so i'm going with it for now
    # Example output 
    #    [[0, 128], [118, 246], [236, 363]]
    spans_boundaries[-1][1] = len(charspans) - 1
    
    # We are tracking the predictions in decoded_preds. Here we set i to 0 so we know
    # we are reading the first prediction
    i = 0
    
    for pred in decoded_preds:
        
        # Get the relations found in the relevant prediction
        relations = rebel_get_triples(pred)
        
        # We need to know which span we are working with - the first 3 predictions
        # are for the first span, etc. which is how we can use current_span_index to 
        # track which span we're busy with
        current_span_index = i // num_return_sequences
        
        # We already have the token start and end for each span, now we want to get
        # the character start and end for each span
        start, end = charspans[spans_boundaries[current_span_index][0]].start, \
                               charspans[spans_boundaries[current_span_index][1]].end

        
        for relation in relations:
            # Find the (first) occurence of the head and tail terms in the relevant 
            # span and return the indices as character spans
            head_span = [i + start for i in hlp.term_charspans(relation["head"], article_text[start:end])]
            tail_span = [i + start for i in hlp.term_charspans(relation["tail"], article_text[start:end])]
            
            # For Label Studio the direction is always expressed as a 'right' relationship.
            # Revisit this for other applications if necessary!
            direction = "right"

            # Some spurious relations seem to occur that are not even in the text for some 
            # reason so check the head and tail both exist at the specified character spans 
            # before proceeding
            if relation["head"].lower() == article_text[head_span[0]:head_span[1]].lower() \
                and relation["tail"].lower() == article_text[tail_span[0]:tail_span[1]].lower():
                new_relation = kg.Relation(
                relation_id=hlp.generate_uid(),
                head_start_char=head_span[0] + chunk[0],
                head_end_char=head_span[1] + chunk[0],
                # Instead of grabbing relation["head"], we use the original text to match case
                head_text=article_text[head_span[0]:head_span[1]],
                tail_start_char=tail_span[0] + chunk[0],
                tail_end_char=tail_span[1] + chunk[0],
                # Instead of grabbing relation["tail"], we use the original text to match case
                tail_text=article_text[tail_span[0]:tail_span[1]],
                relation_type=relation["type"], 
                direction=direction
    )

                # Create a custom key that excludes relation_id
                custom_key = (new_relation.head_start_char, new_relation.head_end_char, new_relation.head_text,
                              new_relation.tail_start_char, new_relation.tail_end_char, new_relation.tail_text,
                              new_relation.relation_type, new_relation.direction)

                # Use the custom key to add the new_relation to the dictionary -
                # duplicates will be ignored
                if custom_key not in unique_relations_dict.keys():
                    unique_relations_dict[custom_key] = new_relation
        
        # Increment the counter and proceed to the next prediction
        i += 1
    
    # Extract the unique Relation objects from the dictionary
    relations_out = list(unique_relations_dict.values())
    if not article.relations:
        article.relations = relations_out
    else:
        article.relations.extend(relations_out)


def flair_get_relations(article: kg.Article, splitter, ner_tagger, rex_tagger, device, restricted = False) -> kg.Article:
    '''
        Extracts the relations from a piece of text, i.e. an article.
        
        Arguments:
            article: Article instance containing article_text
            splitter: splits articles into sentences which is recommended for corpus tagging
            ner_tagger: performs NER as a pre-requisite task
            rex_tagger: performs RE
            device: 'cuda' if GPU is available, else 'cpu'

        Returns:
            Updated Article instance which includes relations.
    '''
    unique_relations_dict = {}
    
    sentences = splitter.split(article.article_text)
    sentence_starts = [sentence.start_position for sentence in sentences]
    ner_tagger.predict(sentences)
    rex_tagger.predict(sentences)
    
    direction = 'right'
    
    relations = []
    for i, sentence in enumerate(sentences):
        for relation in sentence.get_relations():
            # The only relation we are going to use from Flair is alternate_name and experimentation on the train_set
            # indicated that 0.9 would be a reasonable threshold to cut off at. Below that spurious alternate names
            # start appearing in larger quantities
            if not restricted or (restricted and relation.tag == 'alternate_name' and relation.score > 0.999):
                new_relation = kg.Relation(
                        relation_id=hlp.generate_uid(),
                        head_start_char=relation.first.start_position + sentence_starts[i],
                        head_end_char=relation.first.end_position + sentence_starts[i],
                        # Instead of grabbing relation["head"], we use the original text to match case
                        head_text=relation.first.text, 
                        tail_start_char=relation.second.start_position + sentence_starts[i],
                        tail_end_char=relation.second.end_position + sentence_starts[i],
                        # Instead of grabbing relation["tail"], we use the original text to match case
                        tail_text=relation.second.text, 
                        relation_type=relation.tag,
                        direction=direction,
                        score=relation.score)

                # Create a custom key that excludes relation_id
                custom_key = (new_relation.head_start_char, new_relation.head_end_char, new_relation.head_text,
                              new_relation.tail_start_char, new_relation.tail_end_char, new_relation.tail_text,
                              new_relation.relation_type, new_relation.direction)

                # Use the custom key to add the new_relation to the dictionary -
                # duplicates will be ignored
                if custom_key not in unique_relations_dict.keys():
                    unique_relations_dict[custom_key] = new_relation
            
    relations_out = list(unique_relations_dict.values())
    if not article.relations:
        article.relations = relations_out
    else:
        article.relations.extend(relations_out)

# ## Define functions for importing RE from Label Studio
# ## ---------------------------------------------------

def load_rex_from_label_studio(json_file: str, df: pd.DataFrame, annotations: bool) -> list[kg.Article]:
    '''
    Takes in a json file in Label Studio format, and a corresponding DataFrame which includes 
    article Id and AllText, as well as an indicator on whether to read annotations or 
    predictions, and returns a list of Article instances with RE annotations.
    - json_file : path to Label Studio json file
    - df: Dataframe containing Id and AllText columns 
    - annotations: If True read annotations (the HITL annotations), 
      else read predicitons which contains the model predictions
    Returns a list of Article instances containing Relation instances from Label Studio.
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
        
        if annotations:
            main_source = 'annotations'
        else:
            main_source = 'predictions'
                
        entities = {entity['id']: entity for annotation in item[main_source] for entity in annotation['result'] if entity['type'] == 'labels'}
        relations = []
        for annotation in item[main_source]:
            for result in annotation['result']:
                if result['type'] == 'relation':
                    from_entity = entities[result['from_id']]
                    to_entity = entities[result['to_id']]
                    relation = kg.Relation(
                        relation_id=hlp.generate_uid(),
                        head_start_char=from_entity['value']['start'],
                        head_end_char=from_entity['value']['end'],
                        # Workaround required for bug where some text is not populated in the annotation file
                        head_text=article_text[from_entity['value']['start']:from_entity['value']['end']],
                        # head_text=from_entity['value']['text'],
                        tail_start_char=to_entity['value']['start'],
                        tail_end_char=to_entity['value']['end'],
                        # Workaround required for bug where some text is not populated in the annotation file
                        tail_text=article_text[to_entity['value']['start']:to_entity['value']['end']],
                        # tail_text=to_entity['value']['text'],
                        relation_type=result['labels'][0],
                        direction=result['direction'],
                        head_id=result['from_id'],
                        tail_id=result['to_id']
                    )
                    relations.append(relation)
        
        article = kg.Article(
            article_id=str(article_id),  # Ensure the ID is a string if it's not already
            article_text=article_text,
            relations=relations
        )
        
        articles.append(article)
    
    return articles


# ## Define functions for evaluating REX
# ## -----------------------------------

def calc_article_rex_metrics(article_prediction: kg.Article, article_annotations: list, selected: list = []) -> dict:
    '''
    Takes in an Article instance (article_prediction) and a list of annotated Article instances
    (article_annotations). Finds the correct article in article_annotations to compare against.    
    Returns:
        A dict of metric values including:
        - article id
        - precision
        - recall
        - f1
    '''
    
    # Find the right article to compare against in article_annotations
    article_annotations_map = {article.article_id: article for article in article_annotations}
    matched_article = article_annotations_map.get(article_prediction.article_id)
    
    if len(selected) == 0:
        ground_truth_relations = \
            {((relation.head_start_char, relation.head_end_char), \
              (relation.tail_start_char, relation.tail_end_char), \
              (relation.relation_type, relation.direction)) \
              for relation in matched_article.relations}

        predicted_relations = \
            {((relation.head_start_char, relation.head_end_char), \
              (relation.tail_start_char, relation.tail_end_char), \
              (relation.relation_type, relation.direction)) \
              for relation in article_prediction.relations}
    
    elif len(selected) > 0:
        ground_truth_relations = \
            {((relation.head_start_char, relation.head_end_char), \
              (relation.tail_start_char, relation.tail_end_char), \
              (relation.relation_type, relation.direction)) \
              for relation in matched_article.relations if relation.relation_type in selected}

        predicted_relations = \
            {((relation.head_start_char, relation.head_end_char), \
              (relation.tail_start_char, relation.tail_end_char), \
              (relation.relation_type, relation.direction)) \
              for relation in article_prediction.relations if relation.relation_type in selected}

    # The sets let us easily get TP, FP, FN
    TP = list(ground_truth_relations & predicted_relations)
    # These are ones that are in predicted_relations that are not found in the ground_truth_relations
    FP = list(predicted_relations - ground_truth_relations)
    # These are ones that are in ground_truth_relations but are not found in the predicted_relations
    FN = list(ground_truth_relations - predicted_relations)
    
    # Which then lets us calculate precision, recall and F1 according to the strict exact match method
    p_numerator = len(TP)
    p_denominator = len(TP + FP)
    r_numerator = len(TP)
    r_denominator = len(TP + FN)
    precision = p_numerator / p_denominator if p_denominator != 0 else 0
    recall = r_numerator / r_denominator if r_denominator != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # And we save away the errors for review
    errors = [(pred, "FP") for pred in FP]
    errors.extend([(pred, "FN") for pred in FN])

    # And for completeness also the non-errors
    non_errors = [(pred, "TP") for pred in TP]
    
    return {
        "article_id": article_prediction.article_id,
        "p_numerator": p_numerator,
        "p_denominator":p_denominator,
        "r_numerator": r_numerator,
        "r_denominator": r_denominator,
        "precision": precision, # % of ne's found that are correct
        "recall": recall,       # % of ne's in the corpus found
        "f1": f1,
        "errors": errors,
        "non-errors": non_errors
    }


def calc_corpus_rex_metrics(article_metrics: list[dict]) -> tuple[float, float, float]:
    '''
    Takes in a list of article metrics (output from calc_article_rex_metrics()) and returns
    overall precision, recall and f1 score for the corpus of articles.
    '''
    precision = np.mean([article['precision'] for article in article_metrics])
    recall = np.mean([article['recall'] for article in article_metrics])
    f1 = np.mean([article['f1'] for article in article_metrics])
    return (precision, recall, f1)


def populate_inverse_relations(article: kg.Article):
    '''
    Takes in an instance of an article with relations and 
    builds on it by adding available inverse relations.
    '''
    inverse_lookup = hlp.make_lookup_dict_from_df(rebel_flair_overview, 'rebel description', 'inverse description')
    
    if not article.relations:
        unique_relations_dict = {}
    else:
        unique_relations_dict = {
            (relation.head_start_char, relation.head_end_char, 
             relation.head_text, relation.tail_start_char, 
             relation.tail_end_char, relation.tail_text,
             relation.relation_type, relation.direction): relation
            for relation in article.relations
        }
        
    inverse_relations = []
    for relation in article.relations:
        if inverse_lookup.get(relation.relation_type) is not None:
            inverse = kg.Relation(
                relation_id=hlp.generate_uid(),
                head_start_char=relation.tail_start_char,
                head_end_char=relation.tail_end_char,
                head_text=relation.tail_text,
                tail_start_char=relation.head_start_char,
                tail_end_char=relation.head_end_char,
                tail_text=relation.head_text,
                relation_type=inverse_lookup.get(relation.relation_type), 
                direction=relation.direction
            )
            
            custom_key = (inverse.head_start_char, inverse.head_end_char, inverse.head_text,
                          inverse.tail_start_char, inverse.tail_end_char, inverse.tail_text,
                          inverse.relation_type, inverse.direction)

                # Use the custom key to add the new_relation to the dictionary -
                # duplicates will be ignored
            if custom_key not in unique_relations_dict.keys():
                unique_relations_dict[custom_key] = inverse  
                inverse_relations.append(inverse)
    article.relations.extend(inverse_relations)


def flair_to_rebel(article: kg.Article):
    '''
    Takes in an instance of an article with flair relations and 
    swaps out those terms for the corresponding rebel relations.
    '''
    rebel_lookup = hlp.make_lookup_dict_from_df(rebel_flair_overview, 'flair', 'wikidata description mapping')
    for relation in article.relations:
        if rebel_lookup.get(relation.relation_type) is not None:
            relation.relation_type = rebel_lookup.get(relation.relation_type)
       