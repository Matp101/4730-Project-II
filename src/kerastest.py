# %%
import os
import pandas as pd
import sklearn

# %%
#import the data
data = pd.read_csv("../res/Dataset.csv")
data.head(10)

print(data.shape)
print(data['sentiment'].value_counts())

# %%
#suffle the data
data = data.sample(frac=1).reset_index(drop=True)

#split the data
split = 30000

train_reviews = data['review'][:split]
train_sentiments = data['sentiment'][:split]

test_reviews = data['review'][split:].reset_index(drop=True)
test_sentiments = data['sentiment'][split:].reset_index(drop=True)


# print(train_reviews.shape, train_sentiments.shape)
# print(train_reviews[0][0:50]," : ", train_sentiments[0])
# print(test_reviews.shape, test_sentiments.shape)
# print(test_reviews[0][0:50]," : ", test_sentiments[0])

# %% [markdown]
# # NLTK
# 
# pretty sure this is bag of words
# 
# guide is [here](https://realpython.com/python-nltk-sentiment-analysis/)

# %%
# import nltk
# import progressbar as pb

# from nltk.sentiment import SentimentIntensityAnalyzer
# sia = SentimentIntensityAnalyzer()
# bar = pb.ProgressBar()
# results = [
#     'positive' if sia.polarity_scores(review)["compound"] > 0 else 'negative'
#     for review in bar(test_reviews)
# ]

# print("accuracy is ", sklearn.metrics.accuracy_score(test_sentiments, results))

# %% [markdown]
# # Keras
# 
# [Deep learning](https://towardsdatascience.com/an-easy-tutorial-about-sentiment-analysis-with-deep-learning-and-keras-2bf52b9cba91)

# %%
import keras
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical
import re

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(train_reviews.values)
X = tokenizer.texts_to_sequences(train_reviews.values)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

Y = pd.get_dummies(train_sentiments).values
X_test = tokenizer.texts_to_sequences(test_reviews.values)
X_test = pad_sequences(X_test, maxlen=X.shape[1])
Y_test = pd.get_dummies(test_sentiments).values

batch_size = 32
model.fit(X, Y, epochs = 7, batch_size=batch_size, verbose = 2)

score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)

print("score: %.2f" % (score))


