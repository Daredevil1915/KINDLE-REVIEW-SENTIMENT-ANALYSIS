import pandas as pd
import numpy as np
import nltk
import re
import joblib
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

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

data=pd.read_csv("all_kindle_review .csv")
data=data[['reviewText','rating']]
data['rating']=data['rating'].apply(lambda x:0 if x<=2 else 1)
data['tokens']=data['reviewText'].apply(preprocess_text)

X_train,X_test,y_train,y_test=train_test_split(
    data['tokens'],
    data['rating'],
    test_size=0.2,
    random_state=42
)

word2vec_model=Word2Vec(
    sentences=X_train,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

X_train_vectors=[]
for sentence in X_train:
    X_train_vectors.append(
        build_sentence_vector(sentence,word2vec_model,100)
    )

X_test_vectors=[]
for sentence in X_test:
    X_test_vectors.append(
        build_sentence_vector(sentence,word2vec_model,100)
    )

X_train_vectors=np.array(X_train_vectors)
X_test_vectors=np.array(X_test_vectors)

X_train_vectors=np.nan_to_num(X_train_vectors)
X_test_vectors=np.nan_to_num(X_test_vectors)

classifier=LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)
classifier.fit(X_train_vectors,y_train)

predictions=classifier.predict(X_test_vectors)

print("Accuracy:",accuracy_score(y_test,predictions))
print(classification_report(y_test,predictions))

joblib.dump(classifier,"./sentiment_model.pkl")
joblib.dump(word2vec_model,"./word2vec_model.pkl")

print("Model saved successfully")