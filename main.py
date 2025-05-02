import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load your model
model = tf.keras.models.load_model('asl_gesture_model.h5')  # Change to your actual model path

# Constants
IMG_SIZE = 28
LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Mediapipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get bounding box around hand
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            xmin = int(min(x_coords) * w) - 20
            xmax = int(max(x_coords) * w) + 20
            ymin = int(min(y_coords) * h) - 20
            ymax = int(max(y_coords) * h) + 20

            # Ensure bounds
            xmin, ymin = max(xmin, 0), max(ymin, 0)
            xmax, ymax = min(xmax, w), min(ymax, h)

            # Crop and preprocess the hand image
            hand_img = frame[ymin:ymax, xmin:xmax]
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
            hand_img = hand_img / 255.0  # Normalize
            hand_img = hand_img.reshape(1, IMG_SIZE, IMG_SIZE, 3)  # Model expects 3 channels

            # Predict
            pred = model.predict(hand_img)
            class_id = np.argmax(pred)
            confidence = np.max(pred)
            label = LABELS[class_id]

            # Draw result
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("ASL Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
