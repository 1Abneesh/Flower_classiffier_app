# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 02:00:24 2023

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