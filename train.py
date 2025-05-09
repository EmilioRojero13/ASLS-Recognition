import kagglehub
import os
import tensorflow as tf
import numpy as np
import mediapipe as mp
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from collections import Counter
from tensorflow.keras.layers import BatchNormalization

# Download dataset
path = kagglehub.dataset_download("esfiam/american-sign-language-dataset")
train_path = os.path.join(path, "ASL_Gestures_36_Classes/train")

# Label mapping (0-9 + a-z)
label_mapping = {str(i): i for i in range(10)}
counter = 10
for i in range(26):
    ch = chr(97 + i)
    if ch not in ['j', 'z']:
        label_mapping[ch] = counter
        counter += 1

num_classes = len(label_mapping.keys())

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)


# Function to extract landmarks
def extract_landmarks(image):
    image = cv2.resize(image, (224, 224))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        return landmarks
    return None

def augment_landmarks(landmarks):
    augmented = [landmarks]  # original
    mirrored = landmarks.copy()
    for i in range(0, len(mirrored), 3):
        mirrored[i] = 1.0 - mirrored[i]  # mirror x
    augmented.append(mirrored)
    return augmented

# Prepare data with optional mirroring
def prepare_data(data_path):
    X, y = [], []
    for label in os.listdir(data_path):
        label_dir = os.path.join(data_path, label)
        if label not in label_mapping:
            print(f"Skipping unknown label: {label}")
            continue
        numeric_label = label_mapping[label]
        samples_collected = 0
        failed_samples = 0
        for fname in os.listdir(label_dir):
            img_path = os.path.join(label_dir, fname)
            image = cv2.imread(img_path)
            if image is None:
                continue
            landmarks = extract_landmarks(image)

            if landmarks is not None:
                augmented_samples = augment_landmarks(landmarks)
                for sample in augmented_samples:
                    X.append(sample)
                    y.append(numeric_label)
                samples_collected += len(augmented_samples)
            else:
                failed_samples += 1

        
        print(f"{label} with {samples_collected} samples.   Failed samples = {failed_samples}")
                
        if samples_collected < 2:
            print(f"Warning: Not enough samples for label '{label}' ({samples_collected})")
    return np.array(X), to_categorical(y, num_classes), y  # include raw y for stratification

# Load and preprocess
X, y_cat, y_raw = prepare_data(train_path)
print(f"Loaded {len(X)} samples.")

# Filter labels with less than 2 instances to avoid stratify error
label_counts = Counter(y_raw)
print(f"This is label counts: {label_counts}")
valid_indices = [i for i, label in enumerate(y_raw) if label_counts[label] > 1]
X = X[valid_indices]
y_cat = y_cat[valid_indices]
y_raw = [y_raw[i] for i in valid_indices]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y_raw, random_state=42)

# Build model
# model = Sequential([
#     Dense(256, activation='relu', input_shape=(X.shape[1],)),
#     Dropout(0.5),
#     Dense(128, activation='relu'),
#     Dropout(0.3),
#     Dense(34, activation='softmax')
# ])


model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(num_classes, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stop])

# Save
model.save('asl_gesture_model_mediapipe.h5')
