#ICI ON BUILD NOTRE API
import pandas as pd

#on importe nos model
from spot-photo....  import #load_model

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
app.state.model = #load_model()


@app.get('/')
def root():
    return {'greeting': 'Hello',
            'pour faire une recherche': 'tapez : /recherche',
            'avec en param': 'la description de la photo que vous recherchez'}



@app.get('/recherche')
def recherche(query : object):

#return show images
