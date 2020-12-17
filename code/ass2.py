from ner_ml import NERML
import gensim
import sys

def part1(train, test):
    print("Running Part 1: Token only feature")
    # Create NER Instance
    ner = NERML('./data/reuters-train-tab-stripped.en',
    './data/gold_stripped.conll', load_embeddings=False)

    # Selection of only tokens
    sel_feats = ['token']

    # Get Vectorised features
    vec_feats = ner.get_feat_vect(True, sel_feats)

    # Train on all models and get results
    for model in ['NB', 'LR', 'SVM']:
        model = ner.create_classifier(vec_feats, model)
        predictions = ner.set_predictions(model, sel_feats)
        ner.get_prediction_summary(predictions)


def part2(train, test):
    print("Running Part 2: All features Included")
    # Create NER Instance
    ner = NERML('./data/reuters-train-tab-stripped.en',
    './data/gold_stripped.conll', load_embeddings=False)

    # Selection of all features
    sel_feats = ['token', 'ChunkLabel', 'POS-Tag', 'PrevToken', 'FULLCAPS', 'FirstCaps']

    # Get Vectorised features
    vec_feats = ner.get_feat_vect(True, sel_feats)

    # Train on all models and get results
    for model in ['NB', 'LR', 'SVM']:
        model = ner.create_classifier(vec_feats, model)
        predictions = ner.set_predictions(model, sel_feats)
        ner.get_prediction_summary(predictions)


def part3(train, test):
    print("Running Part 3: All features + Word Embeddings")

    print('loading embeddings')
    model = gensim.models.KeyedVectors.load_word2vec_format('./models/GoogleNews-vectors-negative300.bin.gz', binary=True)
    print('loading done')

    # Create NER Instance
    ner = NERML('./data/reuters-train-tab-stripped.en',
    './data/gold_stripped.conll', load_embeddings=model)

    # Selection of all features
    sel_feats = ['token', 'ChunkLabel', 'POS-Tag', 'PrevToken', 'FULLCAPS', 'FirstCaps']

    # Get Vectorised features
    vec_feats = ner.get_feat_vect(True, sel_feats)

    # Train on SVM only and get results
    model = ner.create_classifier(vec_feats, 'SVM')
    predictions = ner.set_predictions(model, sel_feats)
    ner.get_prediction_summary(predictions)


# Running Assignment 2 Stuff
if __name__ == "__main__":
    part_to_run = float(sys.argv[1])
    train_path = sys.argv[2]
    test_path = sys.argv[3]
    
    if part_to_run == 1:
        # Part 1
        # All Models, without new features or embeddings
        part1(train_path, test_path)
    elif part_to_run == 2:
        # Part 2
        # Model without embeddings, but with added features
        part2(train_path, test_path)
    elif part_to_run == 3:
        # Part 3
        # Embeddings also included
        part3(train_path, test_path)
