import matplotlib.pyplot as plt
import mediapipe as mp


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

# Example: Show landmarks from 3 random images in your training dataset
import random

label_folder = random.choice(os.listdir(train_dir))
sample_images = os.listdir(os.path.join(train_dir, label_folder))
for i in range(3):
    sample_img = os.path.join(train_dir, label_folder, sample_images[i])
    print(f"Displaying landmarks for: {sample_img}")
    show_landmarks_on_image(sample_img)
