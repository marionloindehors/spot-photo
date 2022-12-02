from google.colab import auth
from google.cloud import storage
import numpy as np
import pandas as pd

# DEF A FUNCTION TO LOAD X_pred_0_to_1000.CSV depuis bucket
def load_X_pred(url):
    X_pred = np.array(pd.read_csv(url, header=None))
    X_pred = X_pred.astype(np.float32)

    corpus_X_pred = []
    for sentence in X_pred[2]:
        corpus_X_pred.append(sentence)
    return corpus_X_pred

def encode_X_pred(model, corpus_X_pred):
    X_pred_embeddings = model.encode(corpus_X_pred, convert_to_tensor=True)
    return X_pred_embeddings


# DEF A FUNCTION TO LOAD FLICKR30K images**

# DEF A FUNCTION TO LOAD OUR DATASET **
