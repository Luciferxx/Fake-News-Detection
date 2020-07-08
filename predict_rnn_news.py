#Importing the neccessary libraries
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import numpy as np
import re
import string
import pickle
import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
#nltk.download('stopwords')
stopwords_english = stopwords.words('english')

def process_text(text):
    '''
    Function to Pre-Process the text
    text: Raw text to be processed
    '''
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
def predict_news_rnn(text):
    '''
    This function predicts the given text as fake or real
    based on RNN (Many to One Architecture) Model.
    text: Text to be predicted as real or fake
    '''
    # Preprocess the given text
    text = process_text(text)
    #Convert it into a dataframe
    text=pd.Series([text],dtype='string')
    #Maximum length of a row
    MAX_SEQUENCE_LENGTH = 1000
    #Getting the tokenizer on which the dataset was trained
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    #Lets convert the input text to sequence
    X = tokenizer.texts_to_sequences(text.values)
    #Pad if required
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    #Parse it into float values
    X = tf.cast(X,tf.float32)
    #Load the model
    model=tf.keras.models.load_model('saved_model1')
    #Lets some predictions
    prediction = model.predict(X)
    #Last but not the least : Lets get index of the largest prediction
    pred = np.argmax(prediction)
    #Lets see if real or fake
    result=''
    if pred==0:
        result="False News!!"
    else:
        result="Real News!!"
    return result