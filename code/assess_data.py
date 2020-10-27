import numpy as np
import pandas as pd

gold_data = pd.read_csv('./data/gold_stripped.conll', '\t', header=None)

print(gold_data[3].unique())
