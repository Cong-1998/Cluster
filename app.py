# import libraries
import streamlit as st
import malaya
import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud 
import matplotlib.pyplot as plt
from gsdmm import MovieGroupProcess
from topic_modelling import processing
from topic_modelling import token
from topic_modelling import topic_model
from topic_modelling import top_words
from topic_modelling import create_topics_dataframe
from topic_modelling import create_WordCloud

def run():
    file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
    if file_upload is not None:
        data = pd.read_csv(file_upload, encoding='unicode_escape')
        st.write(data)
        wc1, wc2 = processing(data)
        st.image(wc1.to_image())
        st.image(wc2.to_image())

if __name__ == '__main__':
    run()
