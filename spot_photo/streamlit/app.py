#ICI ON BUILD NOTRE INTERFACE
import streamlit as st
import requests

st.write("""Welcome on spot photo !""")

with st.form():
    query = st.text_input('Which photo are you looking for ?', 'Please enter a description')

    model_choice = st.multiselect('Choose your model', ['all-mpnet-base-v2','clip-ViT-B-32' ])

    st.form_submit_button('Search')

results = st.image('./header.png')


params = dict(query=query)

spot_photo_api_url = 'https://taxifare.lewagon.ai/predict' #A CHANGER
response = requests.get(spot_photo_api_url, params=params)

prediction = response.json()
