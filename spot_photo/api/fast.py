#ICI ON BUILD NOTRE API
import pandas as pd

#on importe nos model
from spot_photo.ml_logic.model import load_sentence_similarity_model, compute_similarity, embedding_query
from spot_photo.ml_logic.data import load_X_pred, make_corpus, encode_X_pred

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

app.state.model = load_sentence_similarity_model()

app.state.X_pred = load_X_pred(bucket_name = 'bucket_image_flickr30k',
               file_name = 'X_pred_caption_0_to_1000.csv')

app.state.corpus_X_pred = make_corpus(app.state.X_pred)

app.state.X_pred_embeddings = encode_X_pred(app.state.model, app.state.corpus_X_pred)

@app.get('/')
def root():
    return {'greeting': 'Hello',
            'pour faire une recherche': 'tapez : /recherche',
            'avec en param': 'la description de la photo que vous recherchez'}



@app.get('/recherche')
def recherche(query: str, k: int):
    model = app.state.model
    query_embedding = embedding_query(model, query)
    images_names = compute_similarity(query_embedding, app.state.X_pred_embeddings, k=k)
    result = {}
    for i, image in enumerate(images_names) :
        result[image]=f'resultat n° {i+1}'
    return result
