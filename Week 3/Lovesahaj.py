import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from gensim.parsing.preprocessing import remove_stopwords
import pickle
import sys
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def text_processor(df):
    df['text'] = df['text'].apply(lambda x: remove_stopwords(str(x)))
    df['text'] = df['text'].apply(lambda x: str(x).lower().translate(str.maketrans('', '', string.punctuation)))
    df['sentiment'] = df['sentiment'].apply(lambda x: str(x).lower())
    df['text'] = df['text'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
    return df


if __name__ == '__main__':
    filename = ''
    if len(sys.argv) < 2:
        print('Run again with data frame name as argument.')
        exit()
    else:
        filename = sys.argv[1]

    data_frame = pd.read_csv(filename)
    data_frame = text_processor(data_frame)
    x_test, y_test = data_frame['text'], data_frame['sentiment']

    SGD_model = pickle.load(open('final_model.sav', 'rb'))
    predictions = SGD_model.predict(x_test)

    matrix = confusion_matrix(y_test, predictions)

    print(matrix)

    plt.figure(figsize=(10,10))
    sns.heatmap(matrix)
    plt.show()

    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))