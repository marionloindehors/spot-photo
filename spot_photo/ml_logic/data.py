from google.oauth2 import service_account
from google.cloud import storage
import numpy as np
import pandas as pd

# DEF A FUNCTION TO LOAD X_pred_0_to_1000.CSV depuis bucket
def load_X_pred(bucket_name = 'bucket_image_flickr30k',
                file_name = 'X_pred_caption_0_to_1000.csv'):

    credentials = service_account.Credentials.from_service_account_file(
    'possible-aspect-369317-239ac2560a56.json')

    client = storage.Client(credentials=credentials)
    #client = storage.Client('possible-aspect-369317')
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)

    blob.download_to_filename(file_name)
    #Sinon je le ramène en local download

    #url = f'gs://{bucket_name}/{file_name}'

    X_pred = pd.read_csv(file_name, header=None)

    return X_pred

def make_corpus(X_pred):
    corpus_X_pred = []
    for sentence in X_pred[2]:
        corpus_X_pred.append(sentence)
    return corpus_X_pred

def encode_X_pred(model, corpus_X_pred):
    X_pred_embeddings = model.encode(corpus_X_pred, convert_to_tensor=True)
    return X_pred_embeddings


# DEF A FUNCTION TO LOAD FLICKR30K_captions.csv**
def load_data(bucket_name = 'bucket_image_flickr30k',
                file_name = 'flickr30k_captions.csv'):




    credentials = service_account.Credentials.from_service_account_file(
    'possible-aspect-369317-239ac2560a56.json')

    client = storage.Client(credentials=credentials)
    #client = storage.Client('possible-aspect-369317')
    bucket = client.get_bucket(bucket_name)
    if file_name != 'flickr30k_captions.csv' :
        blob = bucket.get_blob(file_name)
        return blob

    blob = bucket.blob(file_name)


    blob.download_to_filename(file_name)
    #Sinon je le ramène en local download

    #url = f'gs://{bucket_name}/{file_name}'

    data = pd.read_csv(file_name, on_bad_lines='skip', delimiter='|')
    print(data['image_name'][55])
    return data


# DEF A FUNCTION TO LOAD OUR DATASET **
