import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Parameters
img_height, img_width = 28, 28
excluded_labels = {'j', 'z'}
batch_size = 128
epochs = 50

def remove_background(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    return cv2.bitwise_and(img, img, mask=mask)

def load_data(data_path):
    images, labels = [], []
    samples_per_label = {}

    for root, _, files in os.walk(data_path):
        label = os.path.basename(root)
        if label in excluded_labels:
            continue
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (img_width, img_height))
            img = remove_background(img)
            images.append(img)
            labels.append(label)
            if label not in samples_per_label:
                samples_per_label[label] = img

    return np.array(images, dtype='float32') / 255.0, np.array(labels), samples_per_label

def show_sample_images(samples):
    labels = sorted(samples.keys())
    plt.figure(figsize=(15, 6))
    for i, label in enumerate(labels):
        plt.subplot(4, 9, i + 1)
        plt.imshow(cv2.cvtColor(samples[label], cv2.COLOR_BGR2RGB))
        plt.title(label)
        plt.axis('off')
    plt.suptitle("Sample Image Per Label (After Background Removal)", fontsize=16)
    plt.tight_layout()
    plt.show()

def encode_labels(y_train_raw, y_test_raw):
    all_labels = sorted(set(y_train_raw) | set(y_test_raw) - excluded_labels)
    label_to_index = {label: idx for idx, label in enumerate(all_labels)}
    y_train = np.array([label_to_index[label] for label in y_train_raw])
    y_test = np.array([label_to_index[label] for label in y_test_raw])
    return to_categorical(y_train, len(label_to_index)), to_categorical(y_test, len(label_to_index)), len(label_to_index)

def build_model(num_classes):
    return Sequential([
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

def main():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("esfiam/american-sign-language-dataset")
    train_path = os.path.join(path, "ASL_Gestures_36_Classes/train")
    test_path = os.path.join(path, "ASL_Gestures_36_Classes/test")

    print("Processing training data...")
    x_train, y_train_raw, sample_images = load_data(train_path)
    print("Processing testing data...")
    x_test, y_test_raw, _ = load_data(test_path)

    print("Encoding labels...")
    y_train_cat, y_test_cat, num_classes = encode_labels(y_train_raw, y_test_raw)

    print("Shuffling training data...")
    x_train, y_train_cat = shuffle(x_train, y_train_cat, random_state=42)

    print("Displaying 1 sample per label...")
    show_sample_images(sample_images)

    print("Building and training model...")
    model = build_model(num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        x_train, y_train_cat,
        validation_data=(x_test, y_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop]
    )

    print("Saving model to asl-cnn-model.h5...")
    model.save("model.h5")
    print("Done.")

if __name__ == "__main__":
    main()
