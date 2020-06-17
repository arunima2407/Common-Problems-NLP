# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:27:34 2020

@author: aruchakr
"""

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


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

"""
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 5
"""


print('Loading word vectors...')
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


print('Loading in comments...')

train = pd.read_csv("train.csv")
sentences = train["question_text"].values
target = train["target"].values

tokenizer = Tokenizer(num_words=20000) #vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) or into a vector where the coefficient for each token could be binary, based on word count, based on tf-idf.
tokenizer.fit_on_texts(sentences) #Updates internal vocabulary based on a list of texts.
sequences = tokenizer.texts_to_sequences(sentences) #Converts a text to a sequence of words (or tokens).

word2idx = tokenizer.word_index #indexing each word from vector list
print('Found %s unique tokens.' % len(word2idx))

data = pad_sequences(sequences,100)
print('Shape of data tensor:', data.shape)


print('Filling pre-trained embeddings...')
num_words = min(20000, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, 300)) #fill array embedding_matrix with 0s with size num_words, embedding_matrix i.e. 20000,50

embedding1=[]
for word, i in word2idx.items():
    if i < 20000:
        embedding1 = word2vec.get(word)
        if embedding1 is not None:
            embedding_matrix[i] = embedding1


embedding_layer = Embedding( #Turns positive integers (indexes) into dense vectors of fixed size.
  num_words,
  300,
  weights=[embedding_matrix],
  input_length=100,
  trainable=False
)



print('Building model...')

# create an LSTM network with a single LSTM
input_ = Input(shape=(100,))
x = embedding_layer(input_)
# x = LSTM(15, return_sequences=True)(x)
x = Bidirectional(LSTM(15, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(input_, output)
model.compile(
  loss='binary_crossentropy',
  optimizer=Adam(lr=0.01),
  metrics=['accuracy'],
)


print('Training model...')
r = model.fit(
  data,
  target,
  batch_size=128,
  epochs=2,
  validation_split=0.2
)

print("Done with the Training")
print("Predictions:\n")

p = model.predict(data)
train['target'] = model.predict(data, verbose=1) #verbose to get logs

model.save("quora.pb")

import csv
train.to_csv("trained.csv",index=False)