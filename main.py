from gensim.models import KeyedVectors
import pandas as pd
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten
import numpy as np

word2vec = KeyedVectors.load("word2vec_100_3_polish.bin")

train_data = pd.read_csv("train/train.tsv", sep='\t', names=["label", "text"])
test_data = pd.read_csv("test-A/in.tsv", sep='\t', names=["text"])

train_texts = train_data['text'].values
train_labels = train_data['label'].values
test_texts = test_data['text'].values

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

max_sequence_length = max(max(len(seq) for seq in train_sequences), max(len(seq) for seq in test_sequences))
train_sequences_padded = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_sequences_padded = pad_sequences(test_sequences, maxlen=max_sequence_length)

embedding_dim = 100
vocab_size = len(tokenizer.word_index) + 1
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in tokenizer.word_index.items():
    if word in word2vec:
        embedding_matrix[i] = word2vec[word]

model = Sequential()
model.add(Embedding(input_dim=vocab_size,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_sequence_length,
                    trainable=False))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_sequences_padded, train_labels, epochs=10, batch_size=32, validation_split=0.2)

predictions = model.predict(test_sequences_padded)
predictions = (predictions > 0.5).astype(int).flatten()

output = pd.DataFrame({'predicted_label': predictions})
output.to_csv("out.tsv", sep='\t', index=False)
