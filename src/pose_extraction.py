import cv2
import os
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

RAW_VIDEO_BASE_DIR = 'data/raw_videos'
OUTPUT_BASE_DIR = 'data/extracted_keypoints'

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

def extract_pose_from_video(video_path):
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
    return np.array(keypoints_all)

def process_all_videos():
    for dance_name in os.listdir(RAW_VIDEO_BASE_DIR):
        dance_path = os.path.join(RAW_VIDEO_BASE_DIR, dance_name)
        output_path = os.path.join(OUTPUT_BASE_DIR, dance_name)

        if not os.path.isdir(dance_path):
            continue

        os.makedirs(output_path, exist_ok=True)

        for filename in os.listdir(dance_path):
            if filename.endswith('.mp4'):
                video_path = os.path.join(dance_path, filename)
                npy_path = os.path.join(output_path, filename.replace('.mp4', '.npy'))

                if os.path.exists(npy_path):
                    print(f"[{dance_name}] Skip {filename}, already processed.")
                    continue

                print(f"[{dance_name}] Processing {filename}...")
                keypoints = extract_pose_from_video(video_path)
                np.save(npy_path, keypoints)
                print(f"[{dance_name}] Saved to {npy_path}")


if __name__ == "__main__":
    process_all_videos()
