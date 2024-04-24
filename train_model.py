# train_model.py

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from load_data import load_data

# Wyłączenie ostrzeżeń oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Definicja ścieżek
processed_dir = 'processed_spectrograms'

# Wczytywanie danych
X, y, labels_map = load_data(processed_dir)
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, stratify=y, random_state=69)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=420)

X_train = np.load('processed_spectrograms/X_train.npy')
y_train = np.load('processed_spectrograms/y_train.npy')
X_val = np.load('processed_spectrograms/X_val.npy')
X_test = np.load('processed_spectrograms/X_test.npy')
y_val = np.load('processed_spectrograms/y_val.npy')
y_test = np.load('processed_spectrograms/y_test.npy')


# Liczba klas
num_classes = len(labels_map)

# Budowa modelu
# Accuracies:
# 50-100-200 50 epochs = 0.4199
# 100-200-400 50 epochs = 0.3799
# 32-64-128 50 epochs = 0.3100
# 50-100-200 100= 0.5400
model = Sequential([
    Input(shape=(128, 128, 1)),
    Conv2D(50, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(100, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(200, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(200, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])



# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Ewaluacja modelu
val_loss, val_accuracy = model.evaluate(X_val, y_val)
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Plotting the learning and test loss curves
plt.figure(figsize=(10, 5))

# Plotting the training loss
plt.plot(history.history['loss'], label='Training Loss')

# Plotting the validation loss
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Learning and Test Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Saving the plot as a PNG file
plt.savefig('learning_test_loss_curves.png')
plt.show()

# Zapisanie modelu
model.save('trained_model.h5')


#TODO: Learning curve and convolution matrix
