# -*- coding: utf-8 -*-
"""
Character-level RNN Text Generation
Modified to include temperature-based sampling for better text diversity
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# -------------------------------------------------
# Sampling function (ADDED)
# -------------------------------------------------
def sample(preds, temperature=0.8):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

# -------------------------------------------------
# Input text
# -------------------------------------------------
text = "The beautiful girl whom I met last time is very intelligent also"
# text = "The handsome boy whom I met last time is very intelligent also"

chars = sorted(list(set(text)))
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}

# -------------------------------------------------
# Prepare sequences
# -------------------------------------------------
seq_length = 5
sequences = []
labels = []

for i in range(len(text) - seq_length):
    seq = text[i:i + seq_length]
    label = text[i + seq_length]
    sequences.append([char_to_index[c] for c in seq])
    labels.append(char_to_index[label])

X = np.array(sequences)
y = np.array(labels)

X_one_hot = tf.one_hot(X, len(chars))
y_one_hot = tf.one_hot(y, len(chars))

# -------------------------------------------------
# Model configuration
# -------------------------------------------------
text_len = 50

model = Sequential()
model.add(SimpleRNN(
    text_len,
    input_shape=(seq_length, len(chars)),
    activation='relu'
))
model.add(Dense(len(chars), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------------------------
# Model training
# -------------------------------------------------
model.fit(
    X_one_hot,
    y_one_hot,
    epochs=100,
    verbose=1
)

# -------------------------------------------------
# Text generation
# -------------------------------------------------
start_seq = "The handsome boy whom I met "
generated_text = start_seq

for _ in range(text_len):
    x = np.array([[char_to_index[c] for c in generated_text[-seq_length:]]])
    x_one_hot = tf.one_hot(x, len(chars))

    prediction = model.predict(x_one_hot, verbose=0)[0]
    next_index = sample(prediction, temperature=0.1)
    next_char = index_to_char[next_index]

    generated_text += next_char

print("Generated Text:")
print(generated_text)
