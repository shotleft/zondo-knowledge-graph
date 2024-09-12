# Automating knowledge graph construction from news articles

This repo contains all the code that was used to build a knowledge graph from 2081 articles published by News24 on the topic of the _Zondo Commission_ as described in the project report submitted to the University of London. As per agreement with Media24 (who owns the articles), a sample of 30 of the unlocked articles, i.e. those not behind the paywall, is supplied with the code in parquet format (see ```source_data/sample_text_30_unlocked.pq```) which can be read using the Pandas library.

## Acknowledgements

Where very specific ideas have been used that were found in blogs, stackoverflow, and other online resources, these are acknowledged inline with the relevant code. I also made use of Toqan.ai to get 'starter code' for various low-level functions along the way, which were then expanded upon or re-worked before being incorporated into the code base.

## Acronyms in this readme.md

- KG - knowledge graph
- HITL dataset - human-in-the-loop dataset (sample of 30 annotated articles used for evaluation)
- NER - named entity recognition
- CR - coreference resolution
- REX - relation extraction
- EL - entity linking

## ```kg_builder```

This folder contains all the required dataclasses, functions and methods that were used to build and test the knowledge graph. What follows is an overview of the role of each one:

#### ```kg/kg_dataclasses.py```

Includes data classes for structuring key elements:

Information extraction:

- __Article__
    - __NamedEntity__
    - __CrCluster__ (containing __Mention__)
    - __Relation__

Building the KG:

- __KGData__ used to create an instance of a KG
    - __KGEntity__
    - __KGRelation__

 It also includes methods to export to Label Studio to facilitate creation of the HITL dataset.

#### ```kg/kg_processing.py```

Includes all the functions required to build and/or extend a KG, article by article, from an Article class instance. The main algorithm is ```update_kg_from_article``` which calls the sub-algorithm ```process_entity``` which includes performing EL where possible (up to a maximum of 5 retries).

It also includes methods ```prepare_kg_neo4j_files``` and ```prepare_kg_nx_files``` to export the KGData instance to the appropriate formats to be read in by neo4j and the NetworkX library respectively.

#### ```ner/ner_base.py```

Contains the model setup, code used for the initial NER run (Round 1), evaluation options and the option to import from Label Studio

#### ```cr/cr_base.py``` 

Contains the model setup, code used for the initial CR run (Round 1), evaluation options and the option to import from Label Studio

#### ```rex/rex_base.py```

Contains the model setup, code used for the initial REX run (Round 1), evaluation options and the option to import from Label Studio

#### ```cr/ner_cr_enhancements.py```

Contains the enhancements to NER and CR developed for Round 2 descibed in ___5.2	Round 2 – improving baseline model outputs___

#### ```rex/rex_enhancements.py```

Contains the enhancements to REX developed for Round 2 descibed in ___5.2	Round 2 – improving baseline model outputs___ and ___5.4	Round 4 – improving the KG___

#### ```hlp_functions.py```

General functions used across the project.




## ```reference_info```

The file ```reference_info/rebel_flair_selected.csv``` contains the offline work that was done to arrive at the final ontology described in ___4.3	Ontology requirements definition___. 



## Notebooks

The following notebooks reflect the sequential stages of development and testing as described in the accompanying project report:

#### _5.1	Round 1 - component evaluation & selection of project report_

- ```A. round1_initial_model_outputs.ipynb``` was used to obtain the baseline model outputs (and runtimes) for each model tested for the following tasks: NER, CR and RE
- ```B. round1_initial_model_evaluation.ipynb``` was used to evaluate the baseline model outputs against the HITL dataset

#### _5.2	Round 2 – improving baseline model outputs_

- ```C. round2_model_outputs.ipynb``` was used to include several measures designed to improve the results in round 1
- ```D. round2_model_evaluation.ipynb``` was used to evaluate the measures from round 2 against the HITL dataset of 30 articles

#### _5.3	Round 3 – building the first KG on the HITL dataset_
- ```E. round3_first_kg_build.ipynb``` was used to build the first knowledge graph using the 30 sample articles. Evaluation was done offline using the triples from the HITL dataset as the base.

#### _5.4	Round 4 – improving the KG_
- ```F. round4_second_kg_build.ipynb``` was used to build the second knowledge graph using the 30 sample articles again. Evaluation was again done offline using the triples from the HITL dataset as the base.

#### _6.1	The built KG_
- ```G. end_to_end_on_sample_30_unlocked_article.ipynb``` puts it all together. This notebook can be run on the ```source_data/sample_text_30_unlocked.pq``` supplied with the project, and was also used to compile the final knowledge graph on all 2081 articles. Key events and datapoints were output to ```kg_builder.log``` for tracking and troubleshooting.
- ```H. load_data_to_neo4j.ipynb``` small notebook that loads the neo4j text files into neo4j to construct the final KG
- ```I. networkx_analysis.ipynb``` reads the final entities and relations from ```csv_data``` and does an EDA on the graph and its outputs

Additional utility notebooks are included for reference as follows:

- ```0. assemble_source_data.ipynb``` was used to extract the articles from sources and prepare them for further processing (this includes the full set of articles as well as the sample of 30 used for evaluation)
- ```1. compile_ontology_data.ipynb``` was used to update the ontology information when adjustments were made
