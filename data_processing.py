import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Definicja ścieżek
main_dir = 'speech_commands_filtered'
processed_dir = 'processed_spectrograms'

# Tworzenie nowego folderu dla przetworzonych spektrogramów
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

# Iteracja przez podfoldery i przetwarzanie spektrogramów
for subdir in os.listdir(main_dir):
    subdir_path = os.path.join(main_dir, subdir)
    processed_subdir_path = os.path.join(processed_dir, subdir)

    # Tworzenie podfolderu dla przetworzonych spektrogramów, jeśli nie istnieje
    if not os.path.exists(processed_subdir_path):
        os.makedirs(processed_subdir_path)

    for filename in os.listdir(subdir_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(subdir_path, filename)
            # Wczytanie i przetworzenie spektrogramu
            y, sr = librosa.load(file_path, sr=None)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)

            # Normalizacja spektrogramu
            S_dB_norm = (S_dB - np.min(S_dB)) / (np.max(S_dB) - np.min(S_dB))

            # Generowanie nazwy pliku spektrogramu
            spec_filename = filename.replace('.wav', '.png')
            spec_path = os.path.join(processed_subdir_path, spec_filename)
            # Zapisywanie spektrogramu
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_dB_norm, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Normalized Mel-frequency spectrogram')
            plt.tight_layout()
            plt.savefig(spec_path)
            plt.close()
