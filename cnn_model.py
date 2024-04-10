from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Liczba klas (słów) do klasyfikacji
num_classes = 10

# Budowa modelu CNN
model = Sequential([
    # Pierwsza warstwa konwolucyjna
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    # Druga warstwa konwolucyjna
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Trzecia warstwa konwolucyjna
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Spłaszczenie danych i zastosowanie Dropout
    Flatten(),
    Dropout(0.5),  # Dropout zmniejsza ryzyko przeuczenia

    # Warstwy Dense dla klasyfikacji
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  # Softmax do klasyfikacji wieloklasowej
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Podsumowanie modelu
model.summary()
