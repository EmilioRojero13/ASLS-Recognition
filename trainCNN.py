import kagglehub
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import cv2
import numpy as np

# Download the dataset from Kaggle
path = kagglehub.dataset_download("esfiam/american-sign-language-dataset")

# File paths (image folders)
train_path = os.path.join(path, "ASL_Gestures_36_Classes/train")
test_path = os.path.join(path, "ASL_Gestures_36_Classes/test")
print("Train path:", train_path)
print("Test path:", test_path)

# Parameters
img_size = (28, 28)  # Resize all images to 28x28
batch_size = 32
num_classes = 36

# Load image datasets from folders
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=True
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=False
)

# Normalize images
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_ds, validation_data=test_ds, epochs=10)

# Evaluate the model
model.evaluate(test_ds)

# Display and improve visualization of the first image in the training set
for images, labels in train_ds.take(1):
    first_image = images[0].numpy().squeeze()
    first_label = labels[0].numpy()

    print("Shape of first image:", first_image.shape)
    print("One-hot label of first image:", first_label)

# Save the trained model
model.save('asl_gesture_model.h5')