from .kg_dataclasses import NamedEntity, Mention, CRCluster, Relation, Article, KGEntity, KGRelation, KGData, make_articles, ChangeLogDict
from .kg_processing import setup_el_tagger, update_kg_from_article


# We don't want the whole module to be available - only the specified main functions
# https://stackoverflow.com/questions/61278110/excluding-modules-when-importing-everything-in-init-py
del kg_dataclasses
del kg_processing