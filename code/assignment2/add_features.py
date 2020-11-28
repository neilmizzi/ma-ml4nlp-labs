from os import sep
import pandas as pd
from pandas.io.pytables import IndexCol


class AddFeatures:
    def __init__(self, path):
        data = pd.read_csv(path, delimiter='\t', names=['token', 'cat', 'pos-tag', 'ner'])

        # Set Previous and next tokens
        tokens = data['token'].tolist()
        
        prev_token = ['-'] + tokens
        del prev_token[-1]

        next_token = tokens + ['-']
        del next_token[0]

        data = data.assign(PREV_TOKEN=prev_token)
        data = data.assign(NEXT_TOKEN=next_token)

        is_caps = [1 if str(x).isupper() else 0 for x in tokens]
        first_letter_caps = [1 if str(x)[0].isupper() else 0 for x in tokens]

        data = data.assign(CAPS_ALL=is_caps)
        data = data.assign(CAPS_FIRST=first_letter_caps)

        data = data[['token', 'cat', 'pos-tag', 'PREV_TOKEN', 'NEXT_TOKEN', 'CAPS_ALL', 'CAPS_FIRST', 'ner']]

        data.to_csv(path, sep='\t', index=False, header=False)




if __name__ == "__main__":
    AddFeatures('./data/reuters-train-tab-stripped.en/')