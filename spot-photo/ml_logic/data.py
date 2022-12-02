from google.oauth2 import service_account
from google.cloud import storage
import numpy as np
import pandas as pd

# DEF A FUNCTION TO LOAD X_pred_0_to_1000.CSV depuis bucket
def load_X_pred(bucket_name = 'bucket_image_flickr30k',
                file_name = 'X_pred_caption_0_to_1000.csv'):

    credentials = service_account.Credentials.from_service_account_file(
    'wagon-data-1035-b399095159a4.json')

    client = storage.Client(credentials=credentials)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)

    blob.download_to_filename(file_name)
    #Sinon je le ramène en local download

    #url = f'gs://{bucket_name}/{file_name}'

    X_pred = np.array(pd.read_csv(file_name, header=None))
    X_pred = X_pred.astype(np.float32)
    return X_pred

def make_corpus(X_pred):
    corpus_X_pred = []
    for sentence in X_pred[2]:
        corpus_X_pred.append(sentence)
    return corpus_X_pred

def encode_X_pred(model, corpus_X_pred):
    X_pred_embeddings = model.encode(corpus_X_pred, convert_to_tensor=True)
    return X_pred_embeddings


# DEF A FUNCTION TO LOAD FLICKR30K images**

# DEF A FUNCTION TO LOAD OUR DATASET **
