from ner_ml import NERML
from itertools import combinations

def part1():
    print("Running Part 1: Ablation Analysis")
    # Create NER Instance
    ner = NERML('./data/reuters-train-tab-stripped.en',
    './data/gold_stripped.conll', load_embeddings=False)

    # Selection of all features
    sel_feats = ['token', 'ChunkLabel', 'POS-Tag', 'PrevToken', 'NextToken', 'FULLCAPS', 'FirstCaps']

    feature_combs = combinations(sel_feats)
    print(feature_combs)
    # # Get Vectorised features
    # vec_feats = ner.get_feat_vect(True, sel_feats)

    # # Train on all models and get results
    # for model in ['LR']:
    #     model = ner.create_classifier(vec_feats, model)
    #     predictions = ner.set_predictions(model, sel_feats)
    #     ner.get_prediction_summary(predictions)


if __name__ == "__main__":
    part1()
