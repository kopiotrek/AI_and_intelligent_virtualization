import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def save_spectrogram(audio_path, save_dir, subfolder):
    """Generuje i zapisuje spektrogram dla danego pliku audio."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')

        audio_filename = os.path.basename(audio_path).replace('.wav', '')
        plt.title(audio_filename)
        plt.tight_layout()

        save_subfolder = os.path.join(save_dir, subfolder)
        if not os.path.exists(save_subfolder):
            os.makedirs(save_subfolder)

        spec_filename = os.path.basename(audio_path).replace('.wav', '.png')
        save_path = os.path.join(save_subfolder, spec_filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Spektrogram zapisany jako: {save_path}")
    except Exception as e:
        print(f"Nie można wygenerować spektrogramu: {e}")

# Ścieżki
main_dir = 'speech_commands_filtered'
save_dir = 'speech_commands_spectrograms'
subfolder = '27_sheila'

# Iteracja przez pliki od 1_0.wav do 1_49.wav
for i in range(50):
    file_name = f'27_{i}.wav'
    audio_path = os.path.join(main_dir, subfolder, file_name)

    if os.path.isfile(audio_path):
        save_spectrogram(audio_path, save_dir, subfolder)
    else:
        print(f"Plik audio nie został znaleziony: {file_name}")
