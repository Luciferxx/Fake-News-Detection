#Importing the neccessary libraries
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import re
import string
import pickle
import nltk
import pandas as pd
def process_text(text):
    lem = WordNetLemmatizer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)
    # remove old style retext text "RT"
    text = re.sub(r'^RT[\s]+', '', text)
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    #Replace Numerical values with num string
    text = re.sub(r'\d+(\.\d+)?','num', text)
    #Remove white spaces
    text = re.sub(r'\s+',' ', text)
    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)
    # tokenize texts
    tokenizer = WordPunctTokenizer()
    text_tokens = tokenizer.tokenize(text)
    #Removing the punctuations if any
    text_tokens = [word for word in text_tokens if word.isalpha()]
    #To store the final changes
    texts_clean = []
    for word in text_tokens:
        if (word  not in stopwords_english and  # remove stopwords
            word  not in string.punctuation):  # remove punctuation
            # texts_clean.append(word)
            stem_word = lem.lemmatize(word)  # stemming word
            texts_clean.append(stem_word)
    #Converting back to string
    text=' '.join(texts_clean)
    return text
def predict_news_ensemble(text):
    '''
    Takes raw text as input as input
    Predicts whether the news is real or fake
    based on the Ensemble model trained
    '''
    #Lets preprocess the text
    text = process_text(text)
    #Pass the text into a DataFrame
    text=pd.Series([text],dtype='string')
    #Load the vectorizer on which the model was trained
    with open('vectorizer.pickle', 'rb') as handle:
        tfidf = pickle.load(handle)
    #Lets load the model
    with open('Ensemble_model.pickle', 'rb') as handle:
        rf = pickle.load(handle)    
    #Lets vectorise the text
    feature_vector=tfidf.transform(text).todense() 
    #Finally passing it into a dataframe
    x = pd.DataFrame(feature_vector)
    return rf.predict(x)