import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

gold_data = pd.read_csv('./data/gold_stripped.conll', '\t', header=None)

gold_data.columns = ['Word', 'POSTag', 'POS_Partial', 'Named_Entity']

print("Summary:")
print(gold_data.head())
print(f"\n\nUnique Named Entities: {gold_data['Named_Entity'].unique()}")
