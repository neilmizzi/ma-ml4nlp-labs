from code.assignment2.ner_machine_learning import extract_features
import pandas as pd
import numpy as np


class features:
    data = []

    def __init__(self) -> None:
        pass


    def __init__(self, infile) -> None:
        self.extract_features(infile)


    def extract_features(self, infile) -> None:
        with open(infile, 'r', encoding='utf8') as infile:
            for line in infile:
                components = line.rstrip('\n').split()
                if len(components) > 0:
                    token = components[0]
                    feature_dict = {'token':token}
                    self.data.append(feature_dict)
