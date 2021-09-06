# Bag of Words Chatbot Model

import pandas as pd
import nltk 
import numpy as np
import re
from nltk.stem import wordnet # to perform lemmatization
from sklearn.feature_extraction.text import CountVectorizer # to perform bow
from nltk import pos_tag # for parts of speech
from sklearn.metrics import pairwise_distances # to perfrom cosine similarity
from nltk import word_tokenize # to create tokens
from nltk.corpus import stopwords # for stop words

# Download if required
'''
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords') 
'''

# function that performs text normalization steps
def normalization(text):
    
    text = str(text).lower() # text to lower case
    clean_text = re.sub(r'[^ a-z]','',text) # removing special characters
    tokens = nltk.word_tokenize(clean_text) # word tokenizing
    
    lema = wordnet.WordNetLemmatizer() # intializing lemmatization
    lema_words = []
    
    tags_list = pos_tag(tokens,tagset=None) # parts of speech
    
    # Lemmatize all the words in given sentence by assigning correct category
    for token,syntactic_func in tags_list:
        if syntactic_func.startswith('V'):  # Verb
            pos_val = 'v'
        elif syntactic_func.startswith('J'): # Adj
            pos_val = 'a'
        elif syntactic_func.startswith('R'): # Adverb
            pos_val = 'r'
        else:
            pos_val = 'n' # Noun
        lemmatized_word = lema.lemmatize(token, pos_val) # lemmatize
        lema_words.append(lemmatized_word) # appending the lemmatized token
    
    return " ".join(lema_words) # returns the lemmatized tokens as a sentence 


# Function to remove stop words and process (normalize) the corpus
def stopword_(text):      
    lema = wordnet.WordNetLemmatizer() # intializing lemmatization
    lema_words = []
    
    tokens = nltk.word_tokenize(text) # word tokenizing
    tags_list = pos_tag(tokens,tagset=None) # parts of speech
    
    words_to_remove = stopwords.words('english')
    
    # Lemmatize all the words in given sentence by assigning correct category
    for token,syntactic_func in tags_list:
        if token in words_to_remove:
            continue
        if syntactic_func.startswith('V'):  # Verb
            pos_val = 'v'
        elif syntactic_func.startswith('R'): # Adverb
            pos_val = 'r'
        elif syntactic_func.startswith('J'): # Adj
            pos_val = 'a'
        else:
            pos_val = 'n' # Noun
        lemmatized_word = lema.lemmatize(token, pos_val) # lemmatize
        lema_words.append(lemmatized_word) # append the lemmatized token
    
    return " ".join(lema_words) # returns the lemmatized tokens as a sentence 


# function that returns response to query using BOW model
def chat(text):
    s = stopword_(text)
    lemma = normalization(s) # calling the function to perform text normalization
    bow = cv.transform([lemma]).toarray() # applying bow
    cosine_value = 1- pairwise_distances(df_bow,bow, metric = 'cosine' )
    index_value = cosine_value.argmax() # getting index value 
    return df['Text Response'].loc[index_value]


# Preprocess corpus
df = pd.read_excel('queries_and_responses.xlsx')
# Fill in missing responses -> Replaces every null value with the previous row's value
df.ffill(axis = 0,inplace=True)
# normalize whole dataset (user input text)
df['Lemmatized_text'] = df['Context'].apply(normalization)
cv = CountVectorizer() # intializing the count vectorizer
X = cv.fit_transform(df['Lemmatized_text']).toarray()
# returns all the unique word from data 
features = cv.get_feature_names()
df_bow = pd.DataFrame(X, columns = features)

