import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

model = tf.keras.models.load_model('asl_gesture_model_mediapipe.h5')

label_mapping = {str(i): i for i in range(10)}
counter = 10
for i in range(26):
    ch = chr(97 + i)
    if ch not in ['j', 'z']:
        label_mapping[ch] = counter
        counter += 1

inv_label_mapping = {v: k.upper() for k, v in label_mapping.items()}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def extract_landmarks_from_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        return landmarks, hand_landmarks
    return None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks, hand_landmarks = extract_landmarks_from_frame(frame)

    if landmarks is not None:
        input_data = landmarks.reshape(1, -1)
        pred = model.predict(input_data, verbose=0)
        class_id = np.argmax(pred)
        confidence = np.max(pred)
        label = inv_label_mapping.get(class_id, "?")

        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("ASL Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
