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

# Load data
X_train = np.load('processed_spectrograms/X_train.npy')
y_train = np.load('processed_spectrograms/y_train.npy')
X_val = np.load('processed_spectrograms/X_val.npy')
X_test = np.load('processed_spectrograms/X_test.npy')
y_val = np.load('processed_spectrograms/y_val.npy')
y_test = np.load('processed_spectrograms/y_test.npy')

# Number of classes
num_classes = len(labels_map)

# Maximum number of models to try
max_models = 10

# Loop for trying different models
for i in range(max_models):
    # Build model
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
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Record start time
    start_time = time.time()
    
    # Train model
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), verbose=0)
    
    # Record end time
    end_time = time.time()
    
    # Calculate training time
    training_time = end_time - start_time
    
    # Evaluate model
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    # Save model
    model.save(f'model_{i+1}.h5')
    
    # Plot learning and test loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning and Test Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'learning_test_loss_curves_{i+1}.png')
    plt.close()
    
    # Record model performance
    with open('model_performance.txt', 'a') as f:
        f.write(f"Model {i+1}: Validation Accuracy - {val_accuracy}, Test Accuracy - {test_accuracy}, Training Time - {training_time} seconds\n")