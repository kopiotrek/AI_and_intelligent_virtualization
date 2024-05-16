import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2

# Ustawienie zmiennej środowiskowej, aby wyłączyć komunikaty oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Definicja ścieżek
main_dir = 'speech_commands_filtered'
processed_dir = 'processed_spectrograms'

# Tworzenie nowego folderu dla przetworzonych spektrogramów
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)
# Zdefiniujmy rozmiar wejścia
target_size = (128, 128)

# Wczytywanie spektrogramów i ich etykiet do tablic NumPy
X = []  # Lista na spektrogramy
y = []  # Lista na etykiety
labels_map = {}  # Słownik do mapowania nazw folderów na etykiety liczbowe

# Uzyskanie mapowania folderów do etykiet
subdirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
for i, subdir in enumerate(subdirs):
    labels_map[subdir] = i

# Wczytywanie danych
for subdir, label in labels_map.items():
    subdir_path = os.path.join(processed_dir, subdir)
    for spec_filename in os.listdir(subdir_path):
        if spec_filename.endswith('.png'):
            img_path = os.path.join(subdir_path, spec_filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, target_size)
            X.append(img)
            y.append(label)

# Konwersja na tablicę NumPy i skalowanie do [0, 1]
X = np.array(X).astype('float32') / 255.0
X = np.expand_dims(X, axis=-1)  # Dodanie wymiaru kanalów
y = np.array(y)

# One-hot encoding etykiet
y = to_categorical(y, num_classes=len(labels_map))

# Podział na zbiory treningowe, walidacyjne i testowe
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")
# Zapisywanie danych jako pliki .npy
np.save(os.path.join(processed_dir, 'X_train.npy'), X_train)
np.save(os.path.join(processed_dir, 'y_train.npy'), y_train)
np.save(os.path.join(processed_dir, 'X_val.npy'), X_val)
np.save(os.path.join(processed_dir, 'y_val.npy'), y_val)
np.save(os.path.join(processed_dir, 'X_test.npy'), X_test)
np.save(os.path.join(processed_dir, 'y_test.npy'), y_test)
