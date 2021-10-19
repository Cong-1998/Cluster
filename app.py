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
import base64
import gensim
from gensim import corpora, models
from streamlit_tags import st_tags, st_tags_sidebar
from streamlit_player import st_player
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
    name = file_upload.name.replace('.csv', '')
    name = name+"_labelled.csv"

# select cluster
toc.header("Select the number of clusters")
int_val = st.number_input('', min_value=1, max_value=30, value=5, step=1)

# add stopword
list_stop = []
my_expander = st.expander(label='Advanced Setting')
with my_expander:
    st.subheader("[What is stopwords?](#stopwords)")
    keywords = st_tags(
        label='Enter Stopwords:',
        text='Press enter to add more',
        value=[],
        key="aljnf")
    st.write(keywords)
    st.write("*Don't collapse this tab.")
list_stop = keywords

# run the program
result = st.button("Run")
if result:
    wc = []
    ans = []
    st.write("Be patient, need to wait 1 to 2 minutes :smile:")
    wc, ans, final_df = processing(data, gensim, malaya, word_tokenize, np, MovieGroupProcess, pd, WordCloud, int_val, list_stop)
    
    # download labelled file
    file = final_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    linko= f'<a href="data:file/csv;base64,{b64}" download='+name+'>Download csv file</a>'
    st.markdown(linko, unsafe_allow_html=True)
    st.write(ans)
    for i in range(len(wc)):
        st.markdown('Most used words in type '+str(i+1))
        st.image(wc[i].to_image())
        st.write('\n')
st.write('\n')

# how to use
toc.header("How to Use")
st.write("1. Please upload csv file and make sure your data are in first column.")
st.write("*How to convert excel file to [csv file](#csv-file).")
st.write("2. Option: You can expand Advanced Setting tab to add new [stopword](#stopwords).")
st.write("3. Please select the number of clusters.")
st.write("4. Please click the 'Run' button.")
st.write('\n')

st.subheader("CSV file")
st.write('This video will teach you how to convert excel file to csv file.')
# Embed a youtube video
st_player("https://www.youtube.com/watch?v=IBbJzzj5r90")
st.write('\n')

st.subheader("Stopwords")
st.write("Stopwords are the words which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence. For example, the words like ada, apa, dia etc.")
st.write("*Here are [Malay stopwords](https://github.com/Cong-1998/Cluster/blob/main/malay_stopwords.txt).")

toc.generate()
