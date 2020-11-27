from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import gensim
import csv
import pandas as pd
import sys



def extract_embeddings_as_features_and_gold(conllfile,word_embedding_model, added_features):
    '''
    Function that extracts features and gold labels using word embeddings
    
    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    :param added_features: a boolean specifying whether to include new features or not
    :type conllfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    '''
    ### This code was partially inspired by code included in the HLT course, obtained from https://github.com/cltl/ma-hlt-labs/, accessed in May 2020.
    labels = []
    features = []
    
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    for row in csvreader:
        #check for cases where empty lines mark sentence boundaries (which some conll files do).
        if len(row) > 3:
            if row[0] in word_embedding_model:
                vector = word_embedding_model[row[0]]
            else:
                vector = [0]*300
            features.append(vector)
            labels.append(row[-1])
    return features, labels



def is_capitalised(token):
    # CASE TOKEN IS FULL CAPS
    if token.isupper():
        return 2
    # Case Token Has First Letter Capitalised
    elif token[0].isupper():
        return 1
    # case token is lowercase
    else:
        return 0


    # returns previous token, the next token, and level of capitalisation feature


def extract_features_and_labels(trainingfile, added_features):
    data = []
    targets = []
    # TIP: recall that you can find information on how to integrate features here:
    # https://scikit-learn.org/stable/modules/feature_extraction.html

    prev_token = '-'
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]

                # added features
                if added_features:
                    capitalised = is_capitalised(token)
                    phrase_cat = components[1]
                    pos_tag = components[2]

                    feature_dict = {'token': token,
                                    'cat': phrase_cat,
                                    'pos_tag': pos_tag,
                                    'prev_token': prev_token,
                                    'capitalised': capitalised}
                else:
                    feature_dict = {'token': token}
                data.append(feature_dict)
                #gold is in the last column
                targets.append(components[-1])
                # set previous token for the next iteration
                prev_token = token
    return data, targets


def extract_features(inputfile, added_features):
    data = []
    prev_token = '-'
    with open(inputfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                # added features
                if added_features:
                    
                    capitalised = is_capitalised(token)

                    phrase_cat = components[1]
                    pos_tag = components[2]

                    feature_dict = {'token': token,
                                    'cat': phrase_cat,
                                    'pos_tag': pos_tag,
                                    'prev_token': prev_token,
                                    'capitalised': capitalised}
                else:
                    feature_dict = {'token': token}
                data.append(feature_dict)
                # set previous token to current token
                prev_token = token
    return data
    

def create_classifier(train_features, train_targets, modelname, word_to_vec_en):
    if modelname ==  'logreg':
        model = LogisticRegression(max_iter=15000)
        vec = DictVectorizer()
        features_vectorized = vec.fit_transform(train_features)
        model.fit(features_vectorized, train_targets)

    elif modelname == 'NB':
        model = MultinomialNB()
        vec = DictVectorizer()
        features_vectorized = vec.fit_transform(train_features)
        model.fit(features_vectorized, train_targets)

    elif modelname == 'SVM':
        model = SVC(max_iter=15000)
        if not word_to_vec_en:
            vec = DictVectorizer()
            features_vectorized = vec.fit_transform(train_features)
        else:
            vec = train_features
            features_vectorized = train_features

        model.fit(features_vectorized, train_targets)

    else:
        raise Exception()
    
    return model, vec
    
    
def classify_data(model, vec, inputdata, outputfile, added_features, word_to_vec_en):
    if not word_to_vec_en:
        features = extract_features(inputdata, added_features)
        features = vec.transform(features)
    else:
        features = vec
    predictions = model.predict(features)
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(inputdata, 'r'):
        if len(line.rstrip('\n').split()) > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()



def main(argv=None):
    # Get input
    if argv is None:
        argv = sys.argv
    trainingfile = argv[1]
    inputfile = argv[2]
    outputfile = argv[3]

    # Set True to run with added features in args
    added_features = argv[4]
    word_to_vec_en = argv[5]
    
    # Case we use word embeddings
    if word_to_vec_en:
        print('loading embeddings')
        language_model = gensim.models.KeyedVectors.load_word2vec_format('./models/GoogleNews-vectors-negative300.bin.gz', binary=True)
        print('loading done')
        outputfile = outputfile.replace('.conll','_word2vec.conll')
        model_list = ['SVM']
        training_features, gold_labels = extract_embeddings_as_features_and_gold(trainingfile, language_model, added_features)
    
    # If we don't, we run all models and extract the normal features
    else:
        print('Not Using Embeddings')
        training_features, gold_labels = extract_features_and_labels(trainingfile, added_features)
        model_list = ['logreg', 'NB', 'SVM']

    # If we have added features, we adjust the name accordingly
    if added_features:
        print("using extra features")
        outputfile = outputfile.replace('.conll','_added_feats.conll')

    for modelname in model_list:
        print(modelname)
        ml_model, vec = create_classifier(training_features, gold_labels, modelname, word_to_vec_en)
        classify_data(ml_model, vec, inputfile, outputfile.replace('.conll','.' + modelname + '.conll'), added_features, word_to_vec_en)

    
    
if __name__ == '__main__':
    # without added features
    # main(['python', 
    # './data/reuters-train-tab-stripped.en', 
    # './data/gold_stripped.conll', 
    # './data/out.conll', False, False])

    # with added features
    main(['python', 
    './data/reuters-train-tab-stripped.en',
    './data/gold_stripped.conll', 
    './data/out.conll', True, False])
    
    # with word embeddings and no added features
    # main(['python', 
    # './data/reuters-train-tab-stripped.en', 
    # './data/gold_stripped.conll', 
    # './data/out.conll', False, True])
