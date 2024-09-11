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



# ## Define global variables
# ## -----------------------

# Restrict outputs to entities of interest
ENTITIES = {'PERSON', 'GPE', 'LOC', 'EVENT', 'FAC', 'LAW', 'ORG'}

# Essentially we want to avoid getting coreferences for phrases like this one 
# where there is no actual named entity: 'the journalists who were responsible 
# for the worst chapter in the history of South African media since 1994'.
# We'll look if a word forms part of the noun head (or closely associated terms)
# and whether a valid entity is associated with it. If so we'll include it.
# Inspired by https://stackoverflow.com/questions/33289820/noun-phrases-with-spacy
# Supplemented with information at:
# https://downloads.cs.stanford.edu/nlp/software/dependencies_manual.pdf
DEPENDENCIES = {'nsubj', 'nsubjpass', 'dobj', 'iobj', 'pobj', 'csubj', 
                'csubjpass', 'xsubj', 'compound', 'appos', 'nmod'}


# ## Define functions for performing NER & CR with spaCy
# ## ---------------------------------------------------

def get_parse_info(doc: spacy.tokens.doc.Doc, article: kg.Article) -> list:
    '''
    Takes in a spaCy doc and returns the following syntactic parse data points:
        - Token
        - Dependency
        - Part of Speech
        - Entity label (or None)
        - Character span as a tuple (relative to the token's sentence)
        - Sentence number
        - Sentence character span as a tuple (relative to the document)
    See https://downloads.cs.stanford.edu/nlp/software/dependencies_manual.pdf for dependency info
    Example output:
        [{'token': '.',
          'dependency': 'punct',
          'pos': 'PUNCT',
          'entity': None,
          'char_span': (183, 184),
          'sentence_number': 2,
          'sentence_char_span': (71, 72)}, etc.]
    '''
    parse_info = []
    sentence_number = 0  # Initialize sentence number
    
    for sentence in doc.sents:
        sentence_start_char = sentence.start_char
        for token in sentence:
            # Calculate token's character span relative to the sentence
            token_start_relative_to_sentence = token.idx - sentence_start_char
            token_end_relative_to_sentence = token_start_relative_to_sentence + len(token.text)
            
            token_info = {
                'token': token.text,
                'dependency': token.dep_,
                'dependency_head': token.head.text,
                'pos': token.pos_,
                'entity': token.ent_type_ if token.ent_type_ else None,
                'doc_char_span': (token.idx, token.idx + len(token)),
                'sent_char_span': (token_start_relative_to_sentence, token_end_relative_to_sentence),  
                'sentence_number': sentence_number,
                'sentence_span': (sentence.start_char, sentence.end_char)
            }
            parse_info.append(token_info)
        
        sentence_number += 1  # Increment sentence number for the next sentence
    
    article.parse_info = parse_info

def get_sentence_indices(article: kg.Article):
    '''
    Takes in parse_info for an article and returns the start and end characters of each sentence.
    Output format is a dict of sentence_number: (start, end), e.g.
    {1: (0, 134),
     2: (134, 298),
     3: (298, 350), etc.}
    '''
    sentence_indices = dict(set([(token['sentence_number'], token['sentence_span']) for token in article.parse_info]))
    article.sentence_indices = sentence_indices

def get_clean_entities(parse_info: list, article: kg.Article, entities: set = ENTITIES) -> list:
    '''
    Takes in parse_info and returns cleaned entity names. This allows for refinements on entities identified.
    Cleaning scenarios catered for include:
        - including only entities of interest (as defined in the ENTITIES list)
        - including adpositions (ADP) in terms like 'Department of Home Affairs'
        - including possessives (PART) and coordinating conjunctions (CCONJ) in terms like 'Liesel's Cakes and Bakes'
        - including punctuation (PUNCT) in terms like 'PriceWaterhouseCoopers (PWC)'
        - including closing brackets in terms like 'PriceWaterhouseCoopers (PWC)'
        - excluding extraneous determiners (DET) in terms like 'the Sunday Times'
        - excluding extraneous trailing possesives in terms like 'Advocate Zondo's'
        - excluding extraneous closing brackets where no opening bracket was identified
    Tokens that constitute a single entity are joined and punctuation is cleaned up, e.g. removing extra spaces around 
    Sample output:
    [{'text': 'PwC', 'start_char': 68, 'end_char': 71, 'ner_type': 'ORG'},
     {'text': 'Pule Mothibe', 'start_char': 80, 'end_char': 92, 'ner_type': 'PERSON'},
     etc.]
    '''

    # Final output list of clean entities
    clean_entities = []
    # Current group of tokens representing an entity
    current_group = []
    # Current character span of tokens representing an entity
    current_span = [None, None]

    for token in article.parse_info:
        # Check if the token's entity is in the ENTITIES list and whether 
        # its part of speech is one of PROPN, PUNCT, PART or ADP
        # Consecutive tokens make up an entity so this will keep on being 
        # true so long as the next token still matches the criteria
        if token['entity'] in entities and token['pos'] in {'PROPN', 'PUNCT', 'PART', 'ADP', 'CCONJ'}:
            # If this is the first item in the group, set the start of the span
            if current_span[0] is None:
                current_span[0] = token['doc_char_span'][0]
                current_entity_type = token['entity']
            # Update the end of the span to the current item's end - will keep 
            # updating the end span as more entity tokens are found
            current_span[1] = token['doc_char_span'][1]
            current_group.append(token['token'])
        else:
            # If the current group exists
            if current_group:
                # If an opening bracket was included in the entity so far
                # and the next token is a closing bracket then make sure
                # we don't leave it behind
                if '(' in current_group:
                    if token['token'] in {')'}:
                        current_span[1] = token['doc_char_span'][1]
                        current_group.append(token['token'])
                # And finally update the clean_entities list with the clean entity identified
                clean_entities.append({'text': re.sub(r'\s*([\-–—\'])\s*', r'\1', ' '.join(current_group)), 
                                       'start_char': current_span[0], 
                                       'end_char': current_span[1],
                                       'ner_type': current_entity_type})
                # And then reset the current_group and current_span so we can begin the process again
                current_group = []
                current_span = [None, None]
    # Add the last group to the clean_entities list if it's not empty
    if current_group:
        clean_entities.append({'text': re.sub(r'\s*([\-–—\'])\s*', r'\1', ' '.join(current_group)), 
                                       'start_char': current_span[0], 
                                       'end_char': current_span[1],
                                       'ner_type': current_entity_type})

    for entity in clean_entities:
        # Deal with spaces before and after brackets
        entity['text'] =  entity['text'].replace('( ', '(').replace(' )', ')')
        # Deal with apostrophes
        if  entity['text'][-3:] == " 's" or entity['text'][-3:] == " ’s":
            entity['text'] =  entity['text'][0:-3]
            entity['end_char'] = entity['end_char'] - 3
        if  entity['text'][-2:] == "'s" or entity['text'][-2:] == "’s":
            entity['text'] =  entity['text'][0:-2]
            entity['end_char'] = entity['end_char'] - 2
        if  entity['text'][-2:] == "\'" or entity['text'][-2:] == "\’":
            entity['text'] =  entity['text'][0:-2]
            entity['end_char'] = entity['end_char'] - 2
        if  entity['text'][-1:] == "'" or entity['text'][-1:] == "’":
            entity['text'] =  entity['text'][0:-1]
            entity['end_char'] = entity['end_char'] - 1
        

    named_entities = [
        kg.NamedEntity(
            entity_id=hlp.generate_uid(),
            start_char=entity['start_char'],
            end_char=entity['end_char'],
            text=entity['text'],
            ner_type=entity['ner_type']
        )
        for entity in clean_entities
    ]

    article.named_entities = named_entities
    
    return clean_entities


def find_matching_entity_names(char_span_tuples: list[tuple], cleaned_entities: list, 
                               inner_match: bool = True, return_tuples = False) -> set:
    '''
    Inner match scenario (inner_match = True):
    char_span_tuples represent cluster mentions, for example 'my colleague Pearlie Joubert' 
    and cleaned entities is the list of cleaned entities which includes the name 
    'Pearlie Joubert'. The match retrieves 'Pearlie Joubert' as the entity 
    associated with the cluster mention.
    
    Outer match scenario (inner_match = False):
    char_span_tuples represent a partial entity names, for example the head of a mention 
    in a noun phrase like 'Joubert' and cleaned entities is the list of cleaned entities 
    which includes the name 'Pearlie Joubert'. The match retrieves 'Pearlie Joubert' as 
    the entity associated with the partial entity name. Use for completing apposes.    
    '''
    
    matching_entities = []
    # Get cleaned entities that are a match for cluster start and end values
    for mention_start, mention_end in char_span_tuples:
        for entity in cleaned_entities:
            # Check for inner or outer matches
            if inner_match:
                if mention_start <= entity['start_char'] and mention_end >= entity['end_char']:
                    if return_tuples == True:
                        matching_entities.append((entity['text'], (entity['start_char'], entity['end_char'])))
                    else:
                        matching_entities.append(entity['text'])
            else:
                if mention_start >= entity['start_char'] and mention_end <= entity['end_char']:
                    if return_tuples == True:
                        matching_entities.append((entity['text'], (entity['start_char'], entity['end_char'])))
                    else:
                        matching_entities.append(entity['text'])
    # If there are multiple matches get the most complete one, for example if
    # matching_entities contains ['Jacob' and 'Jacob Zuma'] then 'Jacob Zuma'
    # will be returned as the most complete matching entity
    matching_entities = [name for name in matching_entities if not any(name != other_name and name in other_name for other_name in matching_entities)]
    
    return set(matching_entities)   


def get_phrases_with_all_mentions(article_text: str, cluster_char_spans: list[tuple], clean_mentions: set) -> list:
    '''
    Each cluster is composed of tuples of character spans representing the phrases identified 
    in the article as part of the cluster. The clean mentions represent the clean entities found
    within each cluster that are the potential 'main reference(s)' for that cluster. If we can
    find a phrase that contains all the clean mentions this will be the best bet for determining
    which is the main entity for that cluster. However, if this is not possible, we look for the
    first phrase which contains each clean mention. Returns the cluster mention index, relevant 
    character span and phrase text for one or more matching cluster mentions - or None if none
    could be found.
    '''
    # First loop: Check if all mentions are in the span, for example if the mentions
    # (clean entities) are {'PricewaterhouseCoopers (PwC)', 'Pule Mothibe'} then the
    # character span 'aviation-related testimony from PricewaterhouseCoopers (PwC) auditor 
    # Pule Mothibe' will evaluate to True
    for i, (span_start, span_end) in enumerate(cluster_char_spans):
        if all(name in article_text[span_start:span_end] for name in clean_mentions):
            return [{
                'cluster_mention_index': i,
                'cluster_mention_span': (span_start, span_end),
                'cluster_phrase': article_text[span_start:span_end]
            }]
    
    # Failing the above we'll look for individual matching phrases 
    results = []
    
    # Second loop: Check for each mention individually in the spans
    for mention in clean_mentions:
        for i, (span_start, span_end) in enumerate(cluster_char_spans):
            if mention in article_text[span_start:span_end]:
                results.append({
                    'cluster_mention_index': i,
                    'cluster_mention_span': (span_start, span_end),
                    'cluster_phrase': article_text[span_start:span_end]
                })
                break  # Move to the next mention after finding the first match
    
    # Return the results list, which may be empty if no matches are found
    return results if results else None


def get_sentence_for_phrase(char_span: tuple, sentence_indices: dict) -> int:
    '''
    Given a dict of sentences representing start and end of each sentence, e.g.
    {0: (0, 12),
     1: (13, 93),
     2: (93, 227)}
    ... determine which sentence the phrase is contained in, for example the phrase
    represented by the char_span (45, 62) is contained within sentence 1 above.
    '''
    for i in sentence_indices.keys():
        if hlp.tuple_overlap(sentence_indices[i], char_span, bidirectional = False):
            return i
    return None


def get_main_head(phrase_parse_info):
    # This order has been selected based on the 'mainness' of the dependencies as they tend to occur
    # in sentences, see https://downloads.cs.stanford.edu/nlp/software/dependencies_manual.pdf for
    # more detailed examples of each dependency
    ordered_dependencies = ('nsubj', 'dobj', 'csubj', 'nsubjpass', 'csubjpass', 'xsubj', 'iobj', 'pobj', 'poss')
    for dep in ordered_dependencies:
    # Iterate through the dependencies in order and get the first one which will be considered the head
        first_dependency = [token for token in phrase_parse_info if token['dependency'] == dep]
        if first_dependency:
            head = first_dependency[0]
            # If that head is associated with an entity we deem it to be the main entity head
            if head['entity'] != None:
                return (head['token'], head['doc_char_span'])
                break
            # However, there are cases where the head will NOT be the entity, for example in the phrase
            # 'My former colleague Pearlie Joubert of the Sunday Times' the word 'colleague' is a 'nsubj'
            # but the appositional modifier 'Joubert' is the entity associated with this 'nsubj'.
            elif head['entity'] == None:
                apposes = [token for token in phrase_parse_info \
                           if token['dependency_head'] == head['token']\
                           and token['dependency'] == 'appos']
                if apposes:
                    head = apposes[0]
                    return (head['token'], head['doc_char_span'])

                
def get_head_and_conjunctions(head: tuple, phrase_parse_info: list[dict] , clean_entities: list[dict]) -> set:
    '''
    The head has the following format:
    ('Joubert', (29, 36)).
    We search through parse_info to check if there are any conjunctions, for example in the case of 
    'Pearlie Joubert and Stephan Hofstatter' where 'Hofstatter' will have a conjunction relationship
    where the dependency_head will be 'Joubert'. We can thus say that these two entities are referred
    to together in the sentence and so we add 'Hofstatter' to the list of tokens to include when
    fetching the entities associated with this phrase.
    '''
    
    old_len = 0
    tokens_to_include = [head]
    new_len = old_len + 1
    while new_len > old_len:
        old_len = new_len
        conjunction_list = [token for token in phrase_parse_info \
                            if token['dependency_head'] == tokens_to_include[-1][0]\
                            and token['dependency'] == 'conj']
        if conjunction_list:
            additional = conjunction_list[0]
            tokens_to_include.append((additional['token'], additional['doc_char_span']))
            new_len = old_len + 1
    appos_tuples = [token[1] for token in tokens_to_include]
    matching_entities = find_matching_entity_names(appos_tuples, clean_entities, inner_match = False)
    return matching_entities


def get_ner_cr_data(articles: list[kg.Article], cr_tagger, entities: set = ENTITIES, dependencies: set = DEPENDENCIES):
    '''
    Takes in a list of Article instances (articles) and a cr_tagger and performs both NER and CR,
    updating each Article instance with named_entities and cr_clusters.
    '''
    docs = [d for d in cr_tagger.pipe([article.article_text for article in articles], batch_size = 5)]
    outputs = []
    for doc, article in zip(docs, articles):
        doc_sents = [sent.sent.as_doc() for sent in doc.sents]
        
        # Get parse_info  and store in the Article instances in case it is required later
        get_parse_info(doc, article)
        get_sentence_indices(article)
        
        # Get clean_entities and store in the Article instances
        clean_entities = get_clean_entities(article.parse_info, article)
        article.named_entities = [
            kg.NamedEntity(
                entity_id=hlp.generate_uid(),
                start_char=entity['start_char'],
                end_char=entity['end_char'],
                text=entity['text'],
                ner_type=entity['ner_type']
            )
            for entity in clean_entities
        ]
        
        # Coreference clusters require a lot more work including
        # 1) Extract only coreference clusters relevant to our named entities
        # 2) Get the clean entity associated with the many varied coreference mentions
        # 3) Resolve mentions where more than one entity is included
        # The last two are not required for performance of the model, but
        # are essential to work with the coreferences further

        # Get initial clusters identified by the model - these will include clusters that are
        # not relevant to entities as well, for example there may be a cluster around a noun phrase
        # like 'the affadavit signed yesterday'
        raw_clusters = doc._.coref_clusters
        
        # We want to consider all tokens that have been labelled as entities and that belong to one of the 
        # identified dependencies - so we can exclude irrelevant clusters like the one described above
        inclusions = [token['doc_char_span'] for token in article.parse_info if 
                      token['dependency'] in dependencies and token['entity'] in entities]

        # Get only those clusters relevant to the inclusions identified
        valid_clusters = {i: {'char_spans': cluster} for i, cluster in enumerate(raw_clusters) \
                  if any(any(incl_start >= cluster_start and incl_end <= cluster_end for incl_start, incl_end in inclusions) \
                  for cluster_start, cluster_end in cluster)}
        
        # Get the clean entities associated with each valid cluster
        for key, value in valid_clusters.items():
            valid_clusters[key]['mentions'] = find_matching_entity_names(value['char_spans'], clean_entities, inner_match = True)
        
        outputs.append({'doc_sents': doc_sents, 
                        'parse_info': article.parse_info, 
                        'sentence_indices': article.sentence_indices, 
                        'clean_entities': clean_entities, 
                        'valid_clusters': valid_clusters})
        
        # Identify the clusters where more than one mention is associated with the cluster - 
        # these will need to be 'tamed', i.e. an attempt made to identify which is the 
        # main mention, OR potentially a phrase may legitimately contain multiple mentions
        # for example 'PWC and SAA' subsquently referred to as 'they' has two mentions
        # whereas a phrase like 'the Sunday Times journalist Pearlie Joubert' should only
        # return Pearlie Joubert as the main entity of the mention - Sunday Times is an 
        # entity but is merely descriptive in this case.
        clusters_to_tame = [i for i, cluster in valid_clusters.items() if len(cluster['mentions']) > 1] #  
        
        # If there are clusters to tame proceed
        if len(clusters_to_tame) > 0:
            for cluster in clusters_to_tame:
                # Get the phrase(s) that best match the clean mentions in the cluster
                phrases_of_interest = get_phrases_with_all_mentions(article.article_text, \
                                                                    valid_clusters[cluster]['char_spans'], \
                                                                    valid_clusters[cluster]['mentions'])
                
                # For each of those phrases
                for phrase in phrases_of_interest:
                    # Determine in which sentence the phrase is located
                    phrase['sentence'] = get_sentence_for_phrase(phrase['cluster_mention_span'], article.sentence_indices)
                    # And then get just the parse_info for that particular phrase from the sentence
                    phrase['phrase_parse_info'] = [token for token in article.parse_info if token['sentence_number'] == phrase['sentence'] \
                                  and hlp.tuple_overlap(phrase['cluster_mention_span'], token['doc_char_span'], bidirectional = False)]
                    # Get the main head of the phrase
                    phrase['head'] = get_main_head(phrase['phrase_parse_info'])
                    # Get any conjunctions associated with the original head and retrieve matching clean entities
                    if phrase['head'] is not None:
                        phrase['tamed'] = get_head_and_conjunctions(phrase['head'], phrase['phrase_parse_info'], clean_entities)
                # Cater for the case where no tamed mentions were found and set to None
                tame_mentions = [item.get('tamed', None) for item in phrases_of_interest if 'tamed' in item]
                if len(tame_mentions) > 0:
                    # Reduce any duplicate mentions to individual mentions and remove Nones
                    tame_mentions = set.union(*tame_mentions)
                    tame_mentions = {mention for mention in tame_mentions if mention is not None}
                
                # A definitive set of mentions was retrieved - the list tame_mentions will contain a single set
                # for example [{'Pearlie Joubert', 'Stephan Hofstatter'}]
                if len(tame_mentions) == 1:
                    valid_clusters[cluster]['mentions'] = tame_mentions
                # Or no definitive set was found in which case we tag it with the 'AAAmbiguity' keyword
                else:
                    valid_clusters[cluster]['mentions'].add('AAAmbiguous')
        
        
        cr_clusters = []
        for i, cluster in valid_clusters.items():
            mentions = [kg.Mention(
                                mention_id=hlp.generate_uid(),
                                start_char=mention[0],
                                end_char=mention[1],
                                text=article.article_text[mention[0]:mention[1]],
                    )
                for mention in cluster['char_spans']
                ]
            cr_clusters.append(kg.CRCluster(cluster_id = hlp.generate_uid(), mentions = mentions, resolved_text = cluster['mentions']))
        article.cr_clusters = cr_clusters
        
    return outputs
