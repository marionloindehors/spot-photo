#ICI ON BUILD NOTRE INTERFACE
import streamlit as st
import requests

with st.form():
    query = st.text_input('Which photo are you looking for ?', 'Please enter your description')

    st.form_submit_button('Search')

params = dict(query= query)

spot_photo_api_url = 'https://taxifare.lewagon.ai/predict'
response = requests.get(spot_photo_api_url, params=params)

prediction = response.json()
