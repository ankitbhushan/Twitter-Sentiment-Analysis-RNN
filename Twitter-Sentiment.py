import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras import layers

tweets = []
labels = []
num_sentences = 0
with open('training_cleaned.csv', encoding='utf8') as filename:
    data = csv.reader(filename, delimiter=',')
    for row in data:
        tweets.append(row[5])
        if row[0] == '0':
            labels.append(0)
        else:
            labels.append(1)
        num_sentences += 1

embedding_dim = 100
max_length = 16
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<oov>'
test_portion = 0.1

tokenizer = Tokenizer(oov_token=oov_tok)
tokenizer.fit_on_texts(tweets)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(np.array(tweets))
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)

split = int(len(tweets) * test_portion)
x_train = padded[split:]
y_train = labels[split:]
x_val = padded[:split]
y_val = labels[:split]
vocab_size = len(word_index)
# print(f'Vocabulary Size = {len(word_index)}')
# print(x_train[0])
# print(y_train[0])

# Preparing the GloVe word-embeddings matrix
embedding_index = {}
current_dir = os.getcwd()
glove_dir = os.path.join(current_dir, 'glove.6B')
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf8')
for row in f:
    values = row.split()
    word = values[0]
    word_vector = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = word_vector
f.close()

embedding_matrix = np.zeros((vocab_size+1, embedding_dim))
for word, index in word_index.items():
    vector = embedding_index.get(word)
    if vector is not None:
        embedding_matrix[index] = vector

# print(embedding_matrix[:10])

# DEFINING A MODEL
model = models.Sequential()
model.add(layers.Embedding(vocab_size+1,
                           embedding_dim,
                           input_length=max_length,
                           weights=[embedding_matrix],
                           trainable=False))
model.add(layers.Dropout(0.2))
model.add(layers.Conv1D(64, 5, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=4))
model.add(layers.LSTM(64))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

training_sequences = np.array(x_train)
training_labels = np.array(y_train, dtype='float32')
validation_sequences = np.array(x_val)
validation_labels = np.array(y_val, dtype='float32')

history = model.fit(training_sequences, training_labels,
                    epochs=10,
                    validation_data=(validation_sequences, validation_labels),
                    verbose=2)

print('Training Complete')

train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
epochs = range(len(train_acc))
plt.plot(epochs, train_acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.legend()
plt.show()

plt.figure()

plt.plot(epochs, train_loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.legend()
plt.show()
