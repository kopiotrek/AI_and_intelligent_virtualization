# evaluate_model.py

import tensorflow as tf
import numpy as np

# Ścieżka do zapisanego modelu i danych testowych
model_path = 'trained_model.h5'
X_test_path = 'processed_spectrograms/X_test.npy'
y_test_path = 'processed_spectrograms/y_test.npy'

# Wczytanie modelu
model = tf.keras.models.load_model(model_path)


# Wczytanie danych testowych
X_test = np.load(X_test_path)
y_test = np.load(y_test_path)

# Ewaluacja modelu
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")
