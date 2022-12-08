#ICI ON BUILD NOTRE API
import pandas as pd

#on importe nos model
from spot_photo.ml_logic.model import load_sentence_similarity_model,\
    compute_similarity, embedding_query, list_of_image_path
from spot_photo.ml_logic.data import load_X_pred, make_corpus, encode_X_pred, load_pickle,\
     preprocess_img_emmb

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# 💡 Preload the model to accelerate the predictions
# We want to avoid loading the heavy deep-learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the uvicorn server starts
# Then to store the model in an `app.state.model` global variable accessible across all routes!
# This will prove very useful for demo days

app.state.model_1 = load_sentence_similarity_model(model_name='all-mpnet-base-v2')
app.state.model_2 = load_sentence_similarity_model(model_name='clip-ViT-B-32')

app.state.X_pred_1 = load_pickle(file_name = 'captions_our_dataset_4639_9079.pkl')
app.state.X_pred_2 = load_pickle(file_name='list_img_embed_results_flickr30k31121.pkl')

app.state.corpus_X_pred_1 = make_corpus(app.state.X_pred_1)

app.state.X_pred_1_embeddings_1 = encode_X_pred(app.state.model_1, app.state.corpus_X_pred_1)

app.state.X_pred_1_embeddings_2 = preprocess_img_emmb (app.state.X_pred_2)

@app.get('/')
def root():
    return {'greeting': 'Hello',
            'pour faire une recherche': 'tapez : /recherche',
            'avec en param': 'la description de la photo que vous recherchez'}



@app.get('/recherche')
def recherche(model_choice : str, query: str, k: int):
    if model_choice == 'all-mpnet-base-v2':
        model = app.state.model_1
        query_embedding = embedding_query(model, query)
        images_index = compute_similarity(query_embedding, app.state.X_pred_1_embeddings_1, k=k)
        images_names = list_of_image_path (list_of_index=images_index, data=app.state.X_pred_1)
        result = {}
        for i, image in enumerate(images_names) :
            result[image]=f'resultat n° {i+1}'
        return result
    if model_choice == 'clip-ViT-B-32':
        model = app.state.model_2
        query_embedding = embedding_query(model, query)
        images_index = compute_similarity(query_embedding, app.state.X_pred_1_embeddings_2, k=k)
        images_names = list_of_image_path (list_of_index=images_index, data=app.state.X_pred_2)
        result = {}
        for i, image in enumerate(images_names) :
            result[image]=f'resultat n° {i+1}'
        return result
