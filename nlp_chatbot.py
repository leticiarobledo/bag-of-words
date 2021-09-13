# Bag of Words Chatbot Model

import pandas as pd
import nltk 
import re
# Text preprocessing and BoW model
from nltk.stem import wordnet # to perform lemmatization
from nltk import pos_tag # parts of speech
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances # cosine similarity
from nltk.corpus import stopwords # for stop words
# GUI
import tkinter as tk
from tkinter import Frame

# Download if required
'''
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords') 
'''


# Function that performs text normalization
def normalization(text):
    clean_text = re.sub(r'[^ a-z]','',text.lower()) # remove special characters
    tokens = nltk.word_tokenize(clean_text) # word tokenizing
    tags = pos_tag(tokens, tagset=None) # parts of speech
    
    return classify(tags, [])
    
    
# Function to remove stop words and normalize text
def stopword_normalize(text):      
    tokens = nltk.word_tokenize(text) # word tokenizing
    tags = pos_tag(tokens,tagset=None) # parts of speech
    words_to_remove = stopwords.words('english')
    
    return classify(tags, words_to_remove)
    
    
# Auxiliary function that performs text normalization
def classify(tags, words_to_remove):
    lema = wordnet.WordNetLemmatizer() # intializing lemmatization
    lema_words = []
    
    # Lemmatize all the words in given sentence by assigning correct category
    for token,syntactic_function in tags:
        if token in words_to_remove:
            continue
        if syntactic_function.startswith('V'):  # Verb
            pos_val = 'v'
        elif syntactic_function.startswith('R'): # Adverb
            pos_val = 'r'
        elif syntactic_function.startswith('J'): # Adj
            pos_val = 'a'
        else:
            pos_val = 'n' # Noun
        lemmatized_word = lema.lemmatize(token, pos_val) # lemmatize
        lema_words.append(lemmatized_word) # append the lemmatized token
    
    return " ".join(lema_words) # returns the lemmatized tokens as a sentence 


# Function that returns response to query using BOW model
def chatbot(text):
    clean_text = stopword_normalize(text)
    lemma = normalization(clean_text) 
    # create BoW model
    bow = cv.transform([lemma]).toarray()
    cosine_value = 1 - pairwise_distances(df_bow,bow, metric = 'cosine' )
    # get largest (cosine similarity) value
    index_value = cosine_value.argmax()
    best_match = df['Text Response'].loc[index_value]
    return best_match


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


# Start of GUI confirguration
root = tk.Tk()
root.geometry('500x250+500+200')
root.title('ChatBot')

# Style border
frame = Frame(root)
frame.pack(pady=20)

user_input = tk.Entry(root)
user_input.pack()

# Function to maintain conversation 
def chat(event):
    user_text = user_input.get()
    output.config(text=chatbot(user_text))

user_input.bind("<Return>", chat)
output = tk.Label(root, text='')
output.pack()

tk.mainloop()