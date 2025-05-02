import os
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Set paths to your dataset
train_dir = r"C:\Users\emili\.cache\kagglehub\datasets\esfiam\american-sign-language-dataset\versions\1\ASL_Gestures_36_Classes\train"
test_dir = r"C:\Users\emili\.cache\kagglehub\datasets\esfiam\american-sign-language-dataset\versions\1\ASL_Gestures_36_Classes\test"

# Total number of classes: 0-9 and A-Z
num_classes = 36

# Map folder names (labels) to integers
def label_to_index(label):
    if label.isdigit():
        return int(label)
    else:
        return ord(label.lower()) - ord('a') + 10

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Extract hand landmarks from an image using MediaPipe
def extract_landmarks_from_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    return None

# Process the image dataset and apply optional flipping augmentation
def process_data(image_dir, augment=True):
    features = []
    labels = []
    for label in os.listdir(image_dir):
        label_path = os.path.join(image_dir, label)
        if os.path.isdir(label_path):
            class_idx = label_to_index(label)
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Original image
                landmarks = extract_landmarks_from_image(img)
                if landmarks is not None:
                    features.append(landmarks)
                    labels.append(class_idx)

                # Augmented: flipped image
                if augment:
                    flipped_img = cv2.flip(img, 1)
                    flipped_landmarks = extract_landmarks_from_image(flipped_img)
                    if flipped_landmarks is not None:
                        features.append(flipped_landmarks)
                        labels.append(class_idx)

    return np.array(features), np.array(labels)

# Load and process training and test data
print("Processing training data...")
X_train, y_train = process_data(train_dir, augment=True)
print(f"Training samples: {len(X_train)}")

print("Processing test data...")
X_test, y_test = process_data(test_dir, augment=False)
print(f"Test samples: {len(X_test)}")

# Normalize features
X_train = X_train.astype('float32') / np.max(X_train)
X_test = X_test.astype('float32') / np.max(X_test)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Define a simple neural network (fully connected)
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),  # 21 landmarks x 3 coords
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# ---------- Visualization block START ----------

import matplotlib.pyplot as plt

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

def show_landmarks_on_image(img_path, flip=False):
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found or could not be read.")
        return

    if flip:
        img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Convert BGR back to RGB for display
        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Hand Landmarks")
        plt.show()
    else:
        print("No hand landmarks detected.")

# Show landmarks from 3 random images in your training dataset
import random
print("Displaying sample hand landmarks...")

for i in range(36):
    label_folder = random.choice(os.listdir(train_dir))
    sample_images = os.listdir(os.path.join(train_dir, label_folder))
    sample_img = os.path.join(train_dir, label_folder, sample_images[i])
    print(f"Displaying landmarks for: {sample_img}")
    show_landmarks_on_image(sample_img)

# ---------- Visualization block END ----------

# Evaluate and save
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

model.save("asl_landmarks_model.h5")
print("Model saved as 'asl_landmarks_model.h5'")
