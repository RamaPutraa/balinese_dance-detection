import numpy as np
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def visualize_keypoints(keypoints_path, save_video=False, output_path="preview.avi"):
    keypoints = np.load(keypoints_path)
    print(f"Loaded keypoints: {keypoints.shape}")  # (n_frame, 99)

    # Set up image canvas
    width, height = 640, 480
    blank_img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Buat VideoWriter kalau mau simpan
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    for frame_data in keypoints:
        image = blank_img.copy()

        # Ubah 99 float ke list of landmarks
        landmarks = []
        for i in range(33):  # total 33 keypoint
            x = int(frame_data[i*3] * width)
            y = int(frame_data[i*3+1] * height)
            z = frame_data[i*3+2]
            landmarks.append(
                landmark_pb2.NormalizedLandmark(x=x/width, y=y/height, z=z)
            )

        landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=landmarks)
        mp_drawing.draw_landmarks(
            image, landmark_list, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

        cv2.imshow('Keypoints Preview', image)
        if save_video:
            out.write(image)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    if save_video:
        out.release()
    cv2.destroyAllWindows()
