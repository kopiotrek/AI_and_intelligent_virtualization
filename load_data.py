# load_data.py

import numpy as np
import os
import cv2
from tensorflow.keras.utils import to_categorical
def load_data(processed_dir, target_size=(128, 128)):
    X = []
    y = []
    labels_map = {}

    subdirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    for i, subdir in enumerate(subdirs):
        labels_map[subdir] = i

    for subdir, label in labels_map.items():
        subdir_path = os.path.join(processed_dir, subdir)
        for spec_filename in os.listdir(subdir_path):
            if spec_filename.endswith('.png'):
                img_path = os.path.join(subdir_path, spec_filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, target_size)
                X.append(img)
                y.append(label)

    X = np.array(X).astype('float32') / 255.0
    X = np.expand_dims(X, axis=-1)
    y = np.array(y)
    y = to_categorical(y, num_classes=len(labels_map))

    return X, y, labels_map
