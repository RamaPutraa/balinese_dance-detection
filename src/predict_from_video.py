import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tkinter as tk
from tkinter import filedialog

# Konfigurasi
MAX_FRAMES = 50
NUM_FEATURES = 99
LABELS = ['baris', 'pendet', 'rejang_sari']
MODEL_PATH = 'src/model/best_model.h5'

# Load model
model = load_model(MODEL_PATH)

# Setup Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Fungsi ekstrak keypoints dari video
def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_all = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            keypoints_all.append(keypoints)

    cap.release()

    keypoints_all = np.array(keypoints_all)

    # Normalisasi jumlah frame
    if keypoints_all.shape[0] > MAX_FRAMES:
        keypoints_all = keypoints_all[:MAX_FRAMES]
    elif keypoints_all.shape[0] < MAX_FRAMES:
        keypoints_all = pad_sequences([keypoints_all], maxlen=MAX_FRAMES, dtype='float32', padding='post', truncating='post')[0]

    return keypoints_all

# Fungsi pilih file
def choose_file():
    root = tk.Tk()
    root.withdraw()  # Hide main window
    file_path = filedialog.askopenfilename(
        title="Pilih Video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov")]
    )
    return file_path

# Fungsi prediksi
def predict_from_video(video_path):
    keypoints = extract_keypoints_from_video(video_path)
    keypoints = np.expand_dims(keypoints, axis=0)  # Bentuk (1, 50, 99)

    prediction = model.predict(keypoints)
    predicted_class = np.argmax(prediction)

    print(f"Prediksi: {LABELS[predicted_class]} (confidence: {np.max(prediction)*100:.2f}%)")

# Main
if __name__ == "__main__":
    video_path = choose_file()

    if video_path:
        predict_from_video(video_path)
    else:
        print("Tidak ada file yang dipilih.")
