#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 02:08:15 2020

@author: sss

Naive Bayes Algorithm

Given a review is positive what is the probability an individual word is positive
P(class|data) = (P(data|class) * P(class)) / P(data)
P(class|data) = probability of class given the provided data

P(class) - Probability of a positive or negative review
    P(positive) - number of positive reviews/total number of reviews
    P(negative) - number of negative reviews/total number of reviews
    
P(data) - Probability of the word occuring in the review
    P(word) - number of occurences of word/total number of words
    
P(data|class) - Probability of the word occuring given the review
    P(word|positive) - Number of occurences of word in positive reviews/total number of words in positive reviews
    P(word|negative) - Number of occurences of word in negative reviews/total number of words in negative reviews
    
P(class|data) - Probability of the type of review given the word
    P(positive|word) = (P(word|positive) * P(positive)) / P(word)
    P(negative|word) = (P(word|negative) * P(negative)) / P(word) 

To classify take the naive bayes probabilty for each word in a test review 

Calculate P(positive|word) and P(negative|word) for each word in the review
then sum the totals for the review.  Whichever sum is greater indicates whether
the review is positive or negative.  

Priors determined by proportion of positive reviews to negative reviews streaming

A buffer is required in the to store the words not identified 

Preprocessing performed by nltk word tokenize, loading words into the dict
"""

import nltk
from numpy import array
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

FILENAMES = ["positive_reviews.txt", "negative_reviews.txt"]
    
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def clean_doc(text):
    pos_tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    clean_tokens = [word.lower() for word in pos_tokens if word.isalpha() and word not in stop_words and len(word) > 1]  
    return clean_tokens

def add_doc_to_vocab(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)  
    return tokens
    
def doc_to_line(file, vocab):
    tokens = [w for w in file if w in vocab]
    return ' '.join(tokens)

def process_doc(text, vocab):
    lines = list()
    line = doc_to_line(text, vocab)
    lines.append(line)
    return lines

def predict_sentiment(review, vocab, tokenizer, model):
    #clean
    tokens = clean_doc(review)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    # convert to line
    line = ' '.join(tokens)
    # encode
    encoded = tokenizer.texts_to_matrix([line], mode='freq')
    # prediction
    yhat = model.predict(encoded, verbose=0)
    return round(yhat[0,0])


if __name__ == '__main__':
    index_to_word = Counter()
    vocab = Counter()
    pos_text_list = add_doc_to_vocab(FILENAMES[0], vocab)
    neg_text_list = add_doc_to_vocab(FILENAMES[1], vocab)
    
    full_text_list = pos_text_list + neg_text_list
    # split into training and testing texts
    pos_train, pos_test = train_test_split(pos_text_list, test_size = 0.1, shuffle = False)
    neg_train, neg_test = train_test_split(neg_text_list, test_size = 0.1, shuffle = False)
    
    train = neg_train + pos_train 
    test =  neg_test + pos_test 
    
    train_string = process_doc(train, vocab)

    test_string = process_doc(test, vocab)

    
    full_text_string = neg_train + neg_test + pos_train + pos_test
    
    for key, value in vocab.items():
        index_to_word[value] = key
    
    max_words = 10000
    tokenizer = Tokenizer(num_words = len(vocab))
    tokenizer.fit_on_texts(train)
    Xtrain = tokenizer.texts_to_sequences(train)
    Xtrain = tokenizer.sequences_to_matrix(Xtrain)
    
    tokenizer.fit_on_texts(test)
    Xtest = tokenizer.texts_to_sequences(test)
    Xtest = tokenizer.sequences_to_matrix(Xtest)
    
    ytrain = np.asarray([0 for _ in range(len(neg_train))] + [1 for _ in range(len(pos_train))]).astype('float64').reshape(-1, 1)
    ytest = np.asarray([0 for _ in range(len(neg_test))] + [1 for _ in range(len(pos_test))]).astype('float64').reshape(-1, 1)
    
    # define network
    model = Sequential()
    model.add(Dense(50, input_shape=Xtrain.shape, activation='relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # fit network
    model.fit(Xtrain, ytrain, epochs=50,  verbose=2)
    
    # evaluate
    loss, acc = model.evaluate(Xtest, ytest,verbose=0)
    print('Test Loss: %f' % (acc*100))
    print('Test Accuracy: %f' % (acc*100))
    
    # Make Prediction
    text = 'Best amazing fantastic movie, loved it'
    print(predict_sentiment(text, vocab, tokenizer, model))
    #test negative
    text = 'Terrible, waste of time.'
    print(predict_sentiment(text, vocab, tokenizer, model))
    

            
        
        
    
    
    
    