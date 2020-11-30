import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import gensim
import csv
import pandas as pd
import numpy as np
import sys
from eval import compare_outcome, get_macro_score


class NERML:
    # INIT Module
    # 
    # takes train & test dirs and loads DataFrames
    # Also takes boolean, typeset to False, determining whether to load embeddings or not
    # 
    # Sets train and test DFs, loads Vectoriser, and loads word embeddings (if enabled)
    def __init__(self, train: str, test: str, load_embeddings: bool=False, iter_lim:int = 15000) -> None:
        # Set Iteration Limit to be used by SVM & NB (and possibly other models that need early stopping)
        self.iter_lim = iter_lim

        # List of all features loaded
        self.feature_list = ['token', 'ChunkLabel', 'POS-Tag', 'PrevToken', 
        'NextToken', 'FULLCAPS', 'FirstCaps', 'NERLabel']

        # Load train and test files as Pandas DF
        # Names are set in case files do not have headers. Remove them if they do.
        self.train_file : DataFrame = pd.read_csv(train, sep='\t', 
        names=self.feature_list)
        self.train_file.columns = self.feature_list
        self.train_file = self.train_file.fillna(0)
        self.test_file : DataFrame = pd.read_csv(test, sep='\t', 
        names=self.feature_list)
        self.test_file.columns = self.feature_list
        self.test_file = self.test_file.fillna(0)

        # Set Dict Vectorizer
        self.vec = DictVectorizer()

        # loads Google Word2Vec Word Embeddings (Make sure directory matches)
        # only done if embeddings will be used
        # This process takes its time if set to true
        self.embeddings_loaded : bool = False
        if load_embeddings:
            # Possible future update: Make this snippet a separate function
            # Allowing us to load the word embeddings after initialisation of NER class
            # But also retaining the option of doing it within the init process
            print('loading embeddings')
            self.language_model = gensim.models.KeyedVectors.load_word2vec_format('./models/GoogleNews-vectors-negative300.bin.gz', binary=True)
            print('loading done')
            self.embeddings_loaded = True


    # determines whether to work with train or test based on bool
    def train_or_test_getter(self, is_train:bool) -> DataFrame:
        return self.train_file if is_train else self.test_file


    # Sets the vectorised embeddings to the DF
    def set_embeddings(self, is_train:bool, feat:str) -> None:
        if self.embeddings_loaded:
            features = []
            df = self.train_or_test_getter(is_train)
            tokens = df[feat].to_list()

            # code obtained from provided files
            for token in tokens:
                if token in self.language_model:
                    vector = self.language_model[token]
                else:
                    vector = [0]*300
                features.append(vector)
            if is_train:
                self.train_file['embeddings_'+feat] = features
            else:
                self.test_file['embeddings_'+feat] = features 
        else:
            raise FileNotFoundError("Embeddings not Loaded!")
    

    # Get Embeddings as list
    def get_embeddings(self, is_train:bool, feat:str) -> list:
        self.set_embeddings(is_train, feat)
        if is_train:
            return self.train_file['embeddings_'+feat].to_list()
        else:
            return self.test_file['embeddings_'+feat].to_list()
    
    
    # Returns Vectorised Dict Feats for either Train or Test
    def get_feat_vect(self, is_train:bool, selected_features:list) -> any:
        # If embeddings are going to be used, remove token and PrevToken
        if self.embeddings_loaded:
            if 'token' in selected_features:
                selected_features.remove('token')
            if 'PrevToken' in selected_features:
                selected_features.remove('PrevToken')
        
        df = self.train_or_test_getter(is_train)
        dict_feats = []
        for i in range(df.shape[0]):
            sub_dict = {}
            for feat in selected_features:
                sub_dict[feat] = df.at[i, feat]
            dict_feats.append(sub_dict)
        feats = self.vec.fit_transform(dict_feats) if is_train else self.vec.transform(dict_feats)

        # add word embeddings to feature set if they are loaded
        if self.embeddings_loaded:
            combined_vectors = []
            feats = np.array(feats.toarray())
            # Future fix: Add var which specifies which features will be word embeddings (if enabled)
            embeddings_token = self.get_embeddings(is_train, feat='token')
            embeddings_prevTok = self.get_embeddings(is_train, feat='PrevToken')
            for i, vector in enumerate(feats):
                combined_vector = np.concatenate((vector, embeddings_token[i]))
                combined_vector = np.concatenate((combined_vector, embeddings_prevTok[i]))
                combined_vectors.append(combined_vector)
            feats = combined_vectors
        return feats


    # Creates and Returns a trained model, specified by model_name. Can be LR, NB or SVM
    def create_classifier(self, feats, model_name:str) -> any:
        print(f"Training {model_name}...")
        targets = self.train_file['NERLabel'].to_list()
        if model_name ==  'LR':
            model = LogisticRegression(max_iter=self.iter_lim)
            model.fit(feats, targets)

        elif model_name == 'NB':
            model = MultinomialNB()
            model.fit(feats, targets)

        elif model_name == 'SVM':
            model = SVC(max_iter=self.iter_lim)
            model.fit(feats, targets)
        else:
            raise ValueError("Model is not known, or not implemented")
        return model


    # Given a model, we test against the test data
    def set_predictions(self, model, sel_feats) -> list:
        feats = self.get_feat_vect(is_train=False, selected_features=sel_feats)
        predictions = model.predict(feats)
        return predictions


    # Returns Macro Scores
    def get_performance(self, predictions) -> any:
        return get_macro_score(predictions, self.test_file['NERLabel'].to_list())


    # Provides Confusion Matrix and class-by-class scores
    def get_prediction_summary(self, predictions) -> None:
        compare_outcome(predictions, self.test_file['NERLabel'].to_list())
