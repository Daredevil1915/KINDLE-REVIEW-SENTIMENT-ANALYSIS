import streamlit as st
import numpy as np
import re
import joblib
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

@st.cache_resource
def download_nltk_data():
    try:
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        return False

download_nltk_data()

stop_words=set(stopwords.words("english"))
lemmatizer=WordNetLemmatizer()

def preprocess_text(text):
    text=str(text).lower()
    text=re.sub(r"(http|https|ftp|ssh)://\S+","",text)
    text=re.sub("[^a-z ]+","",text)
    text=BeautifulSoup(text,"lxml").get_text()
    words=text.split()
    return [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

def build_sentence_vector(words,model,vector_size):
    vec=np.zeros(vector_size)
    count=0
    for w in words:
        if w in model.wv:
            vec+=model.wv[w]
            count+=1
    return vec/count if count!=0 else vec

@st.cache_resource
def load_models():
    try:
        classifier=joblib.load("sentiment_model.pkl")
        word2vec_model=joblib.load("word2vec_model.pkl")
        return classifier,word2vec_model,None
    except FileNotFoundError as e:
        return None,None,f"Model file not found: {e}"
    except Exception as e:
        return None,None,f"Error loading models: {e}"

classifier,word2vec_model,error=load_models()

st.set_page_config(
    page_title="Kindle Review Sentiment Analyzer",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .stButton button {
        width: 100%;
        background-color: #FF9900;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #E68A00;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .example-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>", unsafe_allow_html=True)
st.title("📚 Kindle Review Sentiment Analyzer")
st.markdown("**Word2Vec-based Machine Learning Model**")
st.markdown("</div>", unsafe_allow_html=True)

st.divider()

if error:
    st.error(f"⚠️ {error}")
    st.info("Please ensure 'sentiment_model.pkl' and 'word2vec_model.pkl' are in the same directory as this script.")
    st.stop()

with st.expander("ℹ️ How to use this tool", expanded=False):
    st.markdown("""
    1. **Enter a review** in the text area below
    2. **Click 'Analyze Sentiment'** to get the prediction
    3. The model will classify the review as **Positive** or **Negative**
    
    The model uses Word2Vec embeddings and machine learning to understand the sentiment of your review.
    """)

st.markdown("### 💭 Try an example:")
col1,col2=st.columns(2)

with col1:
    if st.button("📖 Positive Example", use_container_width=True):
        st.session_state.review_text="Great product! Highly recommend."

with col2:
    if st.button("📕 Negative Example", use_container_width=True):
        st.session_state.review_text="Poor quality. Not satisfied."

st.divider()

st.markdown("### ✍️ Enter Your Review")

if 'review_text' not in st.session_state:
    st.session_state.review_text=""

user_input=st.text_area(
    "",
    value=st.session_state.review_text,
    height=150,
    placeholder="Example: The story was engaging and well written. I couldn't put it down!",
    key="review_input"
)

char_count=len(user_input)
st.caption(f"Characters: {char_count}")

if st.button("🔍 Analyze Sentiment", use_container_width=True):
    if user_input.strip()=="":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        try:
            tokens=preprocess_text(user_input)
            
            if len(tokens)==0:
                st.warning("⚠️ No meaningful words found in the review. Please enter a more detailed review.")
            else:
                vector=build_sentence_vector(tokens,word2vec_model,100)
                vector=np.array(vector).reshape(1,-1)
                
                if np.all(vector==0):
                    st.warning("⚠️ Unable to process review. Please try a different review with more common words.")
                else:
                    prediction=classifier.predict(vector)[0]
                    
                    try:
                        proba=classifier.predict_proba(vector)[0]
                        confidence=max(proba)*100
                    except:
                        confidence=None
                    
                    st.divider()
                    
                    if prediction==1:
                        st.success("### ✅ Positive Review")
                    else:
                        st.error("### ❌ Negative Review")
                    
                    if confidence:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            st.info("Please try again with a different review.")

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>Powered by Word2Vec and Machine Learning | Built with Streamlit</small>
</div>
""", unsafe_allow_html=True)