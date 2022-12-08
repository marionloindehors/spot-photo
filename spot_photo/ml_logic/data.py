# from google.oauth2 import service_account
from google.cloud import storage
import numpy as np
import pandas as pd
import pickle
import torch
from io import BytesIO

# DEF A FUNCTION TO LOAD X_pred_0_to_1000.CSV depuis bucket
def load_X_pred(
    bucket_name="bucket_image_flickr30k", file_name="X_pred_caption_0_to_1000.csv"
):

    client = storage.Client()
    # client = storage.Client('possible-aspect-369317')
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)

    blob.download_to_filename(file_name)  # Sinon je le ramène en local download

    X_pred = pd.read_csv(file_name, header=None)

    return X_pred


def make_corpus(X_pred):
    corpus_X_pred = []
    for sentence in X_pred:
        corpus_X_pred.append(sentence[1])
    return corpus_X_pred


def encode_X_pred(model, corpus_X_pred):
    X_pred_embeddings = model.encode(corpus_X_pred, convert_to_tensor=False)
    return X_pred_embeddings


# DEF A FUNCTION TO LOAD FLICKR30K_captions.csv**
def load_data(bucket_name="bucket_image_flickr30k", file_name="flickr30k_captions.csv"):

    client = storage.Client()
    # client = storage.Client('possible-aspect-369317')

    bucket = client.get_bucket(bucket_name)
    if file_name != "flickr30k_captions.csv":
        blob = bucket.get_blob(file_name)
        return blob

    blob = bucket.blob(file_name)

    blob.download_to_filename(file_name)
    # Sinon je le ramène en local download

    # url = f'gs://{bucket_name}/{file_name}'

    data = pd.read_csv(file_name, on_bad_lines="skip", delimiter="|")
    return data


# DEF A FUNCTION TO LOAD OUR DATASET **


def load_pickle(file_name="list_img_embed_results_flickr30k31121.pkl"):
    blob = load_data(file_name=file_name)
    images_embedding = pickle.load(BytesIO(blob.download_as_bytes()))  # on récupère la liste de tupple, (nom_image, np_array)
    return images_embedding

def preprocess_tensor (images_embedding) :
    images_features = []
    for n in images_embedding:
        images_features.append(
            torch.from_numpy(n)
            )
    return images_features #liste de tensor


def preprocess_img_emmb (images_embedding):
    # on ne veut que les images features en sous forme de tensor (512, )
    images_features = []
    for n in images_embedding:
        images_features.append(
            torch.from_numpy(
                (n).reshape(
                    512,
                )
            )
        )
    return images_features #liste de tensor

def extend_pickle(pickle_name_1, pickle_name_2):
    #  1 - on prend un pickle existant sur le bucket grace à la function load_data
    #  2 - on ouvre le pickle pour retrouver une liste 1
    #  1 - on prend le 2 em pickle existant  grace à la function load_data
    #  2 - on ouvre le pickle pour retrouver une liste 2
    #  3 - on .extend() la liste 1 avec la liste 2
    #  4 - on recrée un nouveau pickle
    #  5 - on remet le pickle sur le bucket grace à la function upload_file

    blob_1 = load_data(file_name=pickle_name_1)    # 1
    list_file_1 = pickle.load(BytesIO(blob_1.download_as_bytes()))     # 2
    blob_2 = load_data(file_name=pickle_name_2)    # 1
    list_file_2 = pickle.load(BytesIO(blob_2.download_as_bytes()))     # 2
    list_file_1.extend(list_file_2)    # 3
    with open(pickle_name_1, "wb") as f:
       pickle.dump(list_file_1, f)    # 4
    upload_file(pickle_name_1, path_to_file= pickle_name_1) # 5


def upload_file(file_name, path_to_file, bucket_name="bucket_image_flickr30k"):

    client = storage.Client()
    #client = storage.Client.from_service_account_json('possible-aspect-369317-4356a1067d8c.json')
    bucket_name = bucket_name
    bucket = client.get_bucket(bucket_name)

    blob = bucket.blob(file_name)
    blob.upload_from_filename(path_to_file)
