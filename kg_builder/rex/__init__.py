# Round 1
from .rex_base import setup_rex_tagger
from .rex_base import rebel_get_relations
from .rex_base import flair_get_relations
from .rex_base import load_rex_from_label_studio
from .rex_base import calc_article_rex_metrics
from .rex_base import calc_corpus_rex_metrics
from .rex_base import populate_inverse_relations
from .rex_base import flair_to_rebel


# Round 2
from .rex_enhancements import remove_self_relations                  # Remove relations where head & tail are the same
                                                                     # after populating all inverses

# Round 3
from .rex_enhancements import get_main_relations_only                # remove inverses entailed by the ontoloty
from .rex_enhancements import cleanup_alternate_name_pairs           # only alternate_name relations that 
                                                                     # include an entity on both sides
from .rex_enhancements import cleanup_duplicate_alternate_name_pairs # We don't need to have both
                                                                     # NPA: National Prosecuting Authority
                                                                     # and National Prosecuting Authority: NPA
from .rex_enhancements import populate_alt_names_mentions            # alt_names is the 'official' ones like NPA
                                                                     # where alt_mentions is the 'unofficial' ones like Zuma
from .rex_enhancements import populate_clean_relation_texts          # Get full names wherever possible, e.g. 
                                                                     # Jacob Zuma instead of just Zuma
from .rex_enhancements import populate_node_types                    # Prefer types from Wikidata ontology, else use
                                                                     # node frequencies or NER types


# Round 4
from .rex_enhancements import remove_ambiguous_relations             # Remove relations where there is more than one
                                                                     # relation between the same entities
from .rex_enhancements import cleanup_overlapping_relations          # Resolve relations that overlap and get the 
                                                                     # 'best' name reference

# We don't want the whole module to be available - only the specified main functions
# https://stackoverflow.com/questions/61278110/excluding-modules-when-importing-everything-in-init-py
del rex_base
del rex_enhancements