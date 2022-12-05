#ICI ON BUILD NOTRE INTERFACE
import streamlit as st
import requests
from google.oauth2 import service_account
from google.cloud import storage
from PIL import Image
from io import BytesIO


query = st.text_input('Which photo are you looking for ?', 'Please enter your description')

#st.form_submit_button('Search')

results = st.image('./header.png')


params = dict(query=query)


spot_photo_api_url = 'http://127.0.0.1:8000/recherche'
response = requests.get(spot_photo_api_url, params=params).json()


st.markdown(f'{response.keys()}')


credentials = service_account.Credentials.from_service_account_file(
'possible-aspect-369317-b19475afaf02.json')

client = storage.Client(credentials=credentials)
#client = storage.Client('possible-aspect-369317')
bucket_name = 'bucket_image_flickr30k'

bucket = client.get_bucket(bucket_name)


blob_l =[]
for image in response :
    file_name = f"flickr30k_images/{image}"
    blob = bucket.get_blob(file_name)
    blob_l.append(blob)
rows = len(response)
for  x in range(rows):
    blob_n = blob_l[x]
    img = Image.open(BytesIO(blob_n.download_as_bytes()))
    st.image(img)
