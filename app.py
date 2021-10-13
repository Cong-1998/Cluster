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
import gensim
from gensim import corpora, models
from gsdmm import MovieGroupProcess
from topic_modelling import processing
from topic_modelling import token
from topic_modelling import topic_model
from topic_modelling import top_words
from topic_modelling import create_topics_dataframe
from topic_modelling import create_WordCloud

class Toc:
    def __init__(self):
        self._items = []
        self._placeholder = None
    
    def title(self, text):
        self._markdown(text, "h1")

    def header(self, text):
        self._markdown(text, "h2", " " * 2)

    def subheader(self, text):
        self._markdown(text, "h3", " " * 4)

    def placeholder(self, sidebar=True):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()

    def generate(self):
        if self._placeholder:
            self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)
    
    def _markdown(self, text, level, space=""):
        key = "-".join(text.split()).lower()
        st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")

def run():
    # hide menu bar
    st.markdown(""" <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style> """, unsafe_allow_html=True)
    
    # set up layout
    padding = 1
    st.markdown(f""" <style>
        .reportview-container .main .block-container{{
            padding-top: {padding}rem;
            padding-right: {padding}rem;
            padding-left: {padding}rem;
            padding-bottom: {padding}rem;
        }} </style> """, unsafe_allow_html=True)

    # set up title
    st.title("GSDMM Topic Modeling")
    st.write('\n')

    # set up sidebar
    st.sidebar.header("Table of Content")
    toc = Toc()
    toc.placeholder()

    # upload file
    toc.header("Upload csv file")
    file_upload = st.file_uploader("", type=["csv"])
    if file_upload is not None:
        data = pd.read_csv(file_upload, encoding='unicode_escape')
        st.write(data)

    # select cluster
    toc.header("Select the number of clusters")
    int_val = st.number_input('', min_value=1, max_value=30, value=5, step=1)
    result = st.button("Run")
    
    # print word cloud
    if result:
        wc = []
        st.write("Be patient, need to wait 1 to 2 minutes :smile:")
        for i in range(int_val):
            wc.append(processing(data, gensim, malaya, word_tokenize, np, MovieGroupProcess, pd, WordCloud, int_val))
            st.image(wc[i].to_image())
        #st.image(wc2.to_image())
    st.write('\n')

    # how to use
    toc.header("How to Use")
    st.write("Please upload csv file, which contain 1 column only.")
    st.write("Blabla...")

if __name__ == '__main__':
    run()
