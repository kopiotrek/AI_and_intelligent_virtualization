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
max_models = 13

# Parameters list
# batch_size = [10, 20, 50, 60]
optimizers = ['Adadelta','Adafactor','Adagrad','Adam','AdamW','Adamax','Ftrl','Lion','LossScaleOptimizer','Nadam','Optimizer','RMSprop','SGD']

# Loop for trying different models
for i in range(max_models):
    # Build model
    model = Sequential([
        Input(shape=(128, 128, 1)),
        Conv2D(25, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(50, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(100, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(100, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(optimizer=optimizers[i], loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Record start time
    start_time = time.time()
    
    # Train model
    history = model.fit(X_train, y_train, batch_size=50, epochs=50, validation_data=(X_val, y_val))
    
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
    model.save(f'model_{i+1}.h5')

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
    with open('model_performance.txt', 'a') as f:
        f.write(f"Model {i+1}: Batch size - {batch_size[i]}, Validation Accuracy - {val_accuracy}, Test Accuracy - {test_accuracy}, Training Time - {training_time} seconds\n")
