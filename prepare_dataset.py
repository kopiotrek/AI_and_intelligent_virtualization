import os
import soundfile as sf
from datasets import load_dataset

# Załaduj dataset
dataset = load_dataset('speech_commands', 'v0.01')

# Wybrane numery
selected_indices = [5, 18, 11, 8, 1, 17, 19, 9, 26, 27]

# Filtrowanie datasetu
filtered_data = {index: [] for index in selected_indices}
for sample in dataset['train']:
    label = sample['label']
    if label in selected_indices and len(filtered_data[label]) < 50:
        filtered_data[label].append(sample)

# Katalog główny, w którym będą zapisane nagrania
output_dir = "speech_commands_filtered"
os.makedirs(output_dir, exist_ok=True)

# Zapisywanie nagrań do odpowiednich folderów
for index, recordings in filtered_data.items():
    index_dir = os.path.join(output_dir, str(index))
    os.makedirs(index_dir, exist_ok=True)

    for i, recording in enumerate(recordings):
        # Pobierz dane audio i częstotliwość próbkowania
        audio_data, sample_rate = recording['audio']['array'], recording['audio']['sampling_rate']
        # Określenie ścieżki zapisu pliku
        file_path = os.path.join(index_dir, f"{index}_{i}.wav")
        # Zapisywanie pliku audio
        sf.write(file_path, audio_data, sample_rate)
