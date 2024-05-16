import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix

# Załaduj wcześniej wytrenowany model
model = tf.keras.models.load_model('trained_model2.h5')

# Załaduj zbiór testowy (zakładając, że zostały zapisane jako pliki numpy)
X_test = np.load('processed_spectrograms/X_test.npy')
y_test = np.load('processed_spectrograms/y_test.npy')

# Przewidywanie etykiet dla zbioru testowego
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Generowanie macierzy pomyłek
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

# Wizualizacja macierzy pomyłek
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()