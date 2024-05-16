import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
from load_data import load_data
import time
# Suppress OneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Define paths
processed_dir = 'processed_spectrograms'
# Load data
X_train = np.load('processed_spectrograms/X_train.npy')
y_train = np.load('processed_spectrograms/y_train.npy')
X_val = np.load('processed_spectrograms/X_val.npy')
X_test = np.load('processed_spectrograms/X_test.npy')
y_val = np.load('processed_spectrograms/y_val.npy')
y_test = np.load('processed_spectrograms/y_test.npy')
X, y, labels_map = load_data(processed_dir)
# Number of classes
num_classes = len(labels_map)
# Maximum number of models to try
max_models = 1
# Parameters list
# batch_size = [10, 20, 50, 60]
# optimizers = ['RMSprop','SGD']
# layer_1 = [3,5,10]
# layer_2 = [6,10,20]
# layer_3 = [12,20,40]
# layer_4 = [24,40,80]
# layer_5 = [24,40,80]
# layer_6 = [24,40,80]
# density = [24,40,80]
# layer_1 = [10,20,30]
# layer_2 = [20,40,60]
# layer_3 = [40,80,120]
# layer_4 = [40,80,120]
# layer_5 = [40,80,120]
# layer_6 = [40,80,120]
# density = [40,80,120]
# layer_1 = [3,5,10,20,40,80]
# layer_2 = [3,5,10,20,40,80]
# layer_3 = [3,5,10,20,40,80]
# layer_4 = [3,5,10,20,40,80]
# layer_5 = [3,5,10,20,40,80]
# layer_6 = [3,5,10,20,40,80]
# density = [3,5,10,20,40,80]
layer_1 = [3,40,80]
layer_2 = [6,40,80]
layer_3 = [12,40,80]
layer_4 = [24,20,40]
layer_5 = [48,10,20]
layer_6 = [48,5,10]
density = [48,5,10]

# Loop for trying different models
for i in range(max_models):
    # Build model
    model = Sequential([
        Input(shape=(128, 128, 1)),
        Conv2D(layer_1[i], kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(layer_2[i], (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(layer_3[i], (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(layer_4[i], (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(layer_5[i], (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        # Conv2D(layer_6[i], (3, 3), activation='relu', padding='same'),
        # MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(density[i], activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(optimizer='AdamW', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Record start time
    start_time = time.time()
    
    # Train model
    history = model.fit(X_train, y_train, batch_size=50, epochs=100, validation_data=(X_val, y_val))
    
    # Record end time
    end_time = time.time()
    
    # Calculate training time
    training_time = end_time - start_time
    
    # Evaluate model
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    # Przewidywanie etykiet dla zbioru testowego
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    # Generowanie macierzy pomyłek
    conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
    
    # Save model
    model.save(f'model_{i+1}.keras')
    # Wizualizacja macierzy pomyłek
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(f'confiuson_matrix_{i+1}.png')
    plt.close()
    
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
    with open('model_performance.txt', 'a') as f: f.write(f"Model {i+1}: architecture: layer_1 = {layer_1[i]},layer_2 = {layer_2[i]},layer_3 = {layer_3[i]},layer_4 = {layer_4[i]},layer_5 = {layer_5[i]},density = {density[i]}, Batch size - 50, Optimizer - AdamW, Validation Accuracy - {val_accuracy}, Test Accuracy - {test_accuracy}, Training Time - {training_time} seconds\n")