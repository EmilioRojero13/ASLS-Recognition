import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("asl-cnn-model.h5")

# Your label map (exclude j and z)
label_map = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y'
]

# Parameters
img_size = 28
frame_interval = 30  # Analyze 1 frame every X frames
frame_counter = 0

# Start webcam
cap = cv2.VideoCapture(0)
predicted_label = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for mirror effect
    # frame = cv2.flip(frame, 1)

    # Only predict every N frames
    if frame_counter % frame_interval == 0:
        # Preprocess frame
        img = cv2.resize(frame, (img_size, img_size))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Predict
        prediction = model.predict(img)
        predicted_index = np.argmax(prediction)
        predicted_label = label_map[predicted_index]

    # Display prediction on frame
    cv2.putText(frame, f'Prediction: {predicted_label}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show webcam feed
    cv2.imshow('ASL Recognition', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_counter += 1

cap.release()
cv2.destroyAllWindows()