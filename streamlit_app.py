import streamlit as st
from PIL import Image
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
new_df = pd.read_csv('fakee.csv')
new_df = new_df.fillna(' ')
X = new_df.drop('label', axis=1)
y = new_df['label']

# Define stemming function
ps = PorterStemmer()
def stemming(tweet):
    stemmed_tweet = re.sub('[^a-zA-Z]',' ',tweet)
    stemmed_tweet = stemmed_tweet.lower()
    stemmed_tweet = stemmed_tweet.split()
    stemmed_tweet = [ps.stem(word) for word in stemmed_tweet if not word in stopwords.words('english')]
    stemmed_tweet = ' '.join(stemmed_tweet)
    return stemmed_tweet

# Apply stemming function to content column
new_df['tweet'] = new_df['tweet'].apply(stemming)

# Vectorize data
x = new_df['tweet'].values
y = new_df['label'].values
vector = TfidfVectorizer()
vector.fit(x)
x = vector.transform(x)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Fit logistic regression model
model = LogisticRegression()
model.fit(x_train,y_train)


# website
st.title('Fake News📰 Detector')
st.subheader('Machine Learning')
st.text('')
col1,col2= st.columns(spec=2)
image=Image.open('fake_news.png')
with col1:
        st.image(image,width=500)
with col2:
        st.markdown('<div style="text-align:right;">The System aims to tackle the growing problem of mis-information and fake news,which can cause significant damage to individuals ,society and institions.By using machine learning algorithms, like machine learning, the system can acccurately classify news, articles as real or fake based on their content, language, and other factors.</div>',unsafe_allow_html=True)
st.text('')


input_text = st.text_input('Enter news Article')

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.success('The News is Fake')
    else:
        st.success('The News Is Real')      
