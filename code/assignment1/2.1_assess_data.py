import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

np.set_printoptions(precision=3)

# RETRIEVE DATA
gold_data = pd.read_csv('./data/gold_stripped.conll', '\t', header=None)
gold_data.columns = ['Word', 'POS_BIO', 'POS', 'Named_Entity']

stanford_data = pd.read_csv('./data/stanford_out_matched_tokens.conll', '\t', header=None)
stanford_data.columns = ['Word', 'token', 'POS', 'Named_Entity', 'NA1', 'NA2']

spacy_data = pd.read_csv('./data/spacy_out_matched_tokens.conll', '\t', header=None)
spacy_data.columns = ['Word', 'BIO', 'Named_Entity']

# GOLD TOOLS SUMMARY
print("Summary:")
print(gold_data.head())

# UNIQUE NAMED ENTITIES
print(f"\n\nUnique Named Entities (Gold): {gold_data['Named_Entity'].unique()}")
print(f"\n\nUnique Named Entities (Stan): {stanford_data['Named_Entity'].unique()}")
print(f"\n\nUnique Named Entities (Spac): {spacy_data['Named_Entity'].unique()}")
