import kagglehub
import os
import cv2
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Download dataset
path = kagglehub.dataset_download("esfiam/american-sign-language-dataset")
train_path = os.path.join(path, "ASL_Gestures_36_Classes/train")
test_path = os.path.join(path, "ASL_Gestures_36_Classes/test")

# Parameters
img_height, img_width = 28, 28
excluded_labels = {'j', 'z'}

def load_data_recursive(data_path):
    images = []
    labels = []

    for root, dirs, files in os.walk(data_path):
        label = os.path.basename(root)
        if label in excluded_labels:
            continue

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue  # skip unreadable images
                img = cv2.resize(img, (img_width, img_height))
                images.append(img)
                labels.append(label)

    return np.array(images, dtype='float32') / 255.0, np.array(labels)

# Load and preprocess data
x_train, y_train_raw = load_data_recursive(train_path)
x_test, y_test_raw = load_data_recursive(test_path)

# Create label map excluding 'j' and 'z'
all_labels = sorted(set(y_train_raw) | set(y_test_raw) - excluded_labels)
label_to_index = {label: idx for idx, label in enumerate(all_labels)}
num_classes = len(label_to_index)

# Encode labels
y_train = np.array([label_to_index[label] for label in y_train_raw])
y_test = np.array([label_to_index[label] for label in y_test_raw])
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Shuffle training data
x_train, y_train_cat = shuffle(x_train, y_train_cat, random_state=42)

# CNN Model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train model
epochs = 50
batch_size = 128

history = model.fit(
    x_train, y_train_cat,
    validation_data=(x_test, y_test_cat),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stop]
)

# Save the trained model
model.save("asl-cnn-model.h5")
