import os
import cv2
import numpy as np
import tensorflow as tf
import kagglehub
import string
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import mediapipe as mp

path = kagglehub.dataset_download("esfiam/american-sign-language-dataset")

train_path = os.path.join(path, "ASL_Gestures_36_Classes/train")
test_path = os.path.join(path, "ASL_Gestures_36_Classes/test")
print("Train path:", train_path)
print("Test path:", test_path)

label_mapping = {str(i): i for i in range(10)}
label_mapping.update({letter: 10 + i for i, letter in enumerate(string.ascii_lowercase)})

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# Function to extract hand landmarks
def extract_landmarks(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return landmarks
    return None

def prepare_data(folder_path):
    data = []
    labels = []
    for label in os.listdir(folder_path):
        label_dir = os.path.join(folder_path, label)
        if not os.path.isdir(label_dir):
            continue
        label_key = label.strip().lower()
        if label_key not in label_mapping:
            print(f"Warning: label '{label}' not found in label mapping")
            continue
        mapped_label = label_mapping[label_key]

        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            landmarks = extract_landmarks(img)
            if landmarks:
                data.append(landmarks)
                labels.append(mapped_label)

    return np.array(data), np.array(labels)

X_train, y_train = prepare_data(train_path)
X_test, y_test = prepare_data(test_path)

num_classes = 36
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

X_train = np.array(X_train)
X_test = np.array(X_test)

model = Sequential([
    Dense(128, activation='relu', input_shape=(len(X_train[0]),)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

model.save('mediapipe_asl_model.h5')
