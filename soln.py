# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:27:34 2020

@author: aruchakr
"""

#IMPORTING LIBRARIES

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score

import keras.backend as K


#LOADING WORD VECTORS

word2vec = {}
with open(os.path.join('glove.6B.300d.txt'), encoding = "utf-8") as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split() #split at space
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32') #numpy.asarray()function is used when we want to convert input to an array.
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))


#LOADING THE DATASET

train = pd.read_csv("train.csv")
sentences = train["question_text"].values
target = train["target"].values


#TOKENISING EACH SENTENCE FROM THE DATASET

tokenizer = Tokenizer(num_words=20000) #vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) or into a vector where the coefficient for each token could be binary, based on word count, based on tf-idf.
tokenizer.fit_on_texts(sentences) #Updates internal vocabulary based on a list of texts.
sequences = tokenizer.texts_to_sequences(sentences) #Converts a text to a sequence of words (or tokens).


#CREATING AN ARRAY FOR WORD AND THEIR INDEX 
word2idx = tokenizer.word_index #indexing each word from vector list
print('Found %s unique tokens.' % len(word2idx))


#PADDING EACH VECTOR WITH 0 TO ENSURE UNIFORM LENGTH OF 100
data = pad_sequences(sequences,100)
print('Shape of data tensor:', data.shape)


#CREATING EMBEDDING MATRIX
num_words = min(20000, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, 300)) #fill array embedding_matrix with 0s with size num_words, embedding_matrix i.e. 20000,50

embedding1=[]
for word, i in word2idx.items():
    if i < 20000:
        embedding1 = word2vec.get(word)
        if embedding1 is not None:
            embedding_matrix[i] = embedding1


#CREATING EMBEDDING LAYER TO FEED INTO THE LSTM
embedding_layer = Embedding( #Turns positive integers (indexes) into dense vectors of fixed size.
  num_words,
  300,
  weights=[embedding_matrix],
  input_length=100,
  trainable=False
)




#CREATING MODEL
input_ = Input(shape=(100,))
x = embedding_layer(input_)

x = Bidirectional(LSTM(15, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(input_, output)
model.compile(
  loss='binary_crossentropy',
  optimizer=Adam(lr=0.01),
  metrics=['accuracy'],
)


#TRAINING THE MODEL
r = model.fit(
  data,
  target,
  batch_size=128,
  epochs=2,
  validation_split=0.2
)

print("Done with the Training")
print("Predictions:\n")


#PREDICTING THE DATA
train['target'] = model.predict(data, verbose=1) #verbose to get logs


#SAVING THE MODEL
model.save("quora.pb")


#CREATING A NEW CSV FILE WITH THE PREDICTED DATA
import csv
train.to_csv("trained.csv",index=False)
