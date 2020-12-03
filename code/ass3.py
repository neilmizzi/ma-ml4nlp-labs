from ner_ml import NERML
from itertools import compress, product
import pandas as pd
import gensim

print('loading embeddings')
model = gensim.models.KeyedVectors.load_word2vec_format('D:/Projects/ma-ml4nlp-labs//models/GoogleNews-vectors-negative300.bin.gz', binary=True)
print('loading done')


# obtained from https://stackoverflow.com/a/6542458/5161292
def combinations(items):
    sub_list = (set(compress(items, mask)) for mask in product(*[[0,1]]*len(items)))
    return [list(x) for x in sub_list]


def part1():
    # Create NER Instance
    ner = NERML('D:/Projects/ma-ml4nlp-labs/data/reuters-train-tab-stripped.en', 
                'D:/Projects/ma-ml4nlp-labs/data/gold_stripped.conll', 
                load_embeddings=model)
    print("Running Part 1: Ablation Analysis")
    print("Running extensive tests on one system")

    # Selection of all features
    sel_feats = ['token', 'ChunkLabel', 'POS-Tag', 'PrevToken', 'FULLCAPS', 'FirstCaps']

    # Get all possible combinations of features
    feature_combs = combinations(sel_feats)
    del feature_combs[0]

    results_dict = {'variables': [],
                    'precision': [],
                    'recall': [],
                    'f-score': []}

    for comb in feature_combs:
        print(f"Testing combination {comb}")
        # Get Vectorised features
        vec_feats = ner.get_feat_vect(True, comb)

        # Train on all models and get results
        for model in ['LR']:
            model = ner.create_classifier(vec_feats, model)
            predictions = ner.set_predictions(model, comb)
            precision, recall, f_score = ner.get_performance(predictions)

            results_dict['variables'].append(comb)
            results_dict['precision'].append(precision)
            results_dict['recall'].append(recall)
            results_dict['f-score'].append(f_score)

    results = pd.DataFrame().from_dict(results_dict)
    results.to_csv('D:/Projects/ma-ml4nlp-labs/data/ass3_results_ablanal_deep.csv', sep='\t')

part1()