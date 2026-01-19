import streamlit as st
import numpy as np
import re
import joblib
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words=set(stopwords.words('english'))
lemmatizer=WordNetLemmatizer()


def preprocess_text(text):
    text=str(text).lower()
    text=re.sub(r'(http|https|ftp|ssh)://\S+','',text)
    text=re.sub('[^a-z ]+','',text)
    text=BeautifulSoup(text,'lxml').get_text()
    words=text.split()
    cleaned_words=[]
    for word in words:
        if word not in stop_words:
            cleaned_words.append(lemmatizer.lemmatize(word))

    return cleaned_words


def build_sentence_vector(words,model,vector_size):
    sentence_vector=np.zeros(vector_size)
    valid_word_count=0

    for word in words:
        if word in model.wv:
            sentence_vector+=model.wv[word]
            valid_word_count+=1

    if valid_word_count>0:
        sentence_vector=sentence_vector/valid_word_count

    return sentence_vector


classifier=joblib.load("sentiment_model.pkl")
word2vec_model=joblib.load("word2vec_model.pkl")


st.set_page_config(page_title="Kindle Review Sentiment",layout="centered")

st.title("Amazon Kindle Review Sentiment Analysis")
st.write("Enter a Kindle review to predict sentiment using Word2Vec")

user_input=st.text_area("Enter Review")

if st.button("Predict Sentiment"):
    if user_input.strip()=="":
        st.warning("Please enter some text")
    else:
        tokens=preprocess_text(user_input)
        vector=build_sentence_vector(tokens,word2vec_model,100)
        vector=np.array(vector).reshape(1,-1)

        prediction=classifier.predict(vector)[0]

        if prediction==1:
            st.success("Positive Review 👍")
        else:
            st.error("Negative Review 👎")
