import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

VOCAB_SIZE = 10000
MAX_LEN = 250
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 4

(X_train_int, y_train), (X_test_int, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

X_train_padded = pad_sequences(X_train_int, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_int, maxlen=MAX_LEN, padding='post', truncating='post')

print("开始构建 LSTM 模型...")
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),

    Bidirectional(LSTM(64, return_sequences=True)),

    Dropout(0.3),

    Bidirectional(LSTM(64)),

    Dense(1, activation='sigmoid')
])

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("Training begins...")
history = model.fit(
    X_train_padded,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test_padded, y_test)
)

loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
print(f"\nBest Accuracy: {accuracy:.4f}")
print(f"Final Loss: {loss:.4f}")

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'bo-', label='Training Acc')
    plt.plot(epochs_range, val_acc, 'ro-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'bo-', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_history(history)