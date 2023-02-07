# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:09:22 2023

@author: 01abn
"""

import time
import requests
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import plotly.express as px
# from streamlit import dcc


st.set_page_config(
    page_title="Flower species recognition",
    page_icon="🌻🌹",
    layout='wide'
)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lootie_url = "https://assets10.lottiefiles.com/private_files/lf30_cmd8kh2q.json"
lottie_flower = load_lottieurl(lootie_url)

headers = """# How to use VisionAI"""
st.markdown(headers,unsafe_allow_html=True)

col1,col2,col3 = st.columns([1,2,1])
with col2:
    st_lottie(lottie_flower, key="hello",speed=1, loop=True, quality="medium", width=500,height=400)
    
content = """

1. Launch the app: 
    To launch the app, run the script that contains your Streamlit code. This will start a local web server and open the app in a browser.

2. Select an image: 
    In the app, you should have a button or file uploader that allows the user to select an image of a flower. Click the button or select an image from your device to upload it to the app.

3. Select a pretrained CNN model: 
    Before making a prediction, you can select from a list of pretrained CNN models. Choose the model that you want to use to make the prediction.

4. Wait for the prediction: 
    After the image and model are selected, the app will perform the necessary computations and make a prediction about the species of the flower in the image. This may take a few seconds, depending on the complexity of the model and the size of the image.

5. View the prediction: 
    The app will display the prediction results, including the top prediction for the species of the flower and its probability.

6. Repeat the process: 
    If you want to classify another flower species, you can repeat the steps 2-5 to get a prediction for a new image.

**Note:** The app's accuracy is greater than 75%, but there may still be some errors in the predictions. Always verify the predictions with additional sources before taking any action based on them.
"""
st.markdown(content,unsafe_allow_html=True)

header1 = """## Use case diagram of the App"""
st.markdown(header1,unsafe_allow_html=True)


st.image('use_case.png','Note: the app can classify more than 102 distinct species', width=600)
