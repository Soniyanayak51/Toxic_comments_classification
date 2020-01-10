
import pandas as pd
import numpy as np


test = pd.read_csv('/home/soniya51/Toxic Comments/test.csv')
train = pd.read_csv('/home/soniya51/Toxic Comments/train.csv')
train.head()
train.isnull().any(),test.isnull().any()
import keras
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
sentences_train = train["comment_text"]
sentences_test = test["comment_text"]
type(sentences_train[1])

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(sentences_train))
tokenized_train = tokenizer.texts_to_sequences(sentences_train)
tokenized_test = tokenizer.texts_to_sequences(sentences_test)

maxlen = 200
X_t = pad_sequences(tokenized_train, maxlen=maxlen)
X_te = pad_sequences(tokenized_test, maxlen=maxlen)

#200 is the length of each padded array
'''
def divide_chunks(l, n): 
    for i in range(0, len(l), n): 
        yield l[i:i + n] 
def pad(int_comments):
    for single_comment in int_comments:
        if(len(single_comment) > 200):
            single_comment = list(divide_chunks(single_comment, 200))
            n = len(single_comment)
            single_comment[n-1] += [0] * (200 - len(single_comment[n-1]))
        else:
            single_comment += [0] * (200 - len(single_comment))
    return int_comments
'''

inp = Input(shape=(maxlen, ))

get_ipython().system('ls')

MAX_NB_WORDS = 100000    # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 200 # max length of each entry (sentence), including padding
VALIDATION_SPLIT = 0.2   # data for validation (not used in training)
EMBEDDING_DIM = 100      # embedding dimensions for word vectors (word2vec/GloVe)
GLOVE_DIR = "glove.6B."+str(EMBEDDING_DIM)+"d.txt"


word_index = tokenizer.word_index
print((word_index['name']))

embeddings_index = {}
f = open(GLOVE_DIR)
print('Loading GloVe from:', GLOVE_DIR,'...', end='')
for line in f:
    values = line.split()
    word = values[0]
    embeddings_index[word] = np.asarray(values[1:], dtype='float32')
f.close()
print("Done.\n Proceeding with Embedding Matrix...", end="")

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print(" Completed!")

embedding_layer = Embedding(len(word_index) + 1,
                           EMBEDDING_DIM,
                           weights = [embedding_matrix],
                           input_length = MAX_SEQUENCE_LENGTH,
                           trainable=False,
                           name = 'embeddings')
embedded_sequences = embedding_layer(inp)

x = LSTM(60, return_sequences=True,name='lstm_layer')(embedded_sequences)

x = GlobalMaxPool1D()(x)

x = Dropout(0.1)(x)

x = Dense(50, activation="relu")(x)

x = Dropout(0.1)(x)

preds = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=preds)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

batch_size = 32
epochs = 2
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

print(preds)

y_test = model.predict([X_te], batch_size=1024, verbose=1)
sample_submission = pd.read_csv("../soniya51/Downloads/sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv('submission.csv', index=False)

results = pd.read_csv('submission.csv')

print(len(results))

results.head()




