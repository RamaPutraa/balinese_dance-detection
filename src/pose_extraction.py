import cv2
import os
import numpy as np
import mediapipe as mp
import torch
import time

# Cek ketersediaan GPU
print(f"CUDA tersedia: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU yang terdeteksi: {torch.cuda.get_device_name(0)}")
    print(f"Jumlah GPU: {torch.cuda.device_count()}")
    # Set environment variable untuk MediaPipe
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    print("Tidak ada GPU terdeteksi. Menggunakan CPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Cek apakah OpenCV memiliki dukungan CUDA (aman)
opencv_has_cuda = False
try:
    count = cv2.cuda.getCudaEnabledDeviceCount()
    opencv_has_cuda = count > 0
    if opencv_has_cuda:
        print(f"OpenCV CUDA terdeteksi dengan {count} perangkat")
    else:
        print("OpenCV tidak memiliki dukungan CUDA")
except:
    print("OpenCV tidak dikompilasi dengan dukungan CUDA")

# Inisialisasi MediaPipe Pose dengan parameter yang dioptimalkan
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,  # Model paling akurat
    enable_segmentation=False,  # Nonaktifkan fitur yang tidak perlu
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

RAW_VIDEO_BASE_DIR = 'data/raw_videos'
OUTPUT_BASE_DIR = 'data/extracted_keypoints'

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

def extract_pose_from_video(video_path):
    """Ekstrak keypoints pose dari video"""
    cap = cv2.VideoCapture(video_path)
    keypoints_all = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Memproses {total_frames} frame dengan {fps} FPS")
    
    # Untuk pengukuran kecepatan
    start_time = time.time()
    frame_count = 0
    
    # Gunakan GPU acceleration untuk MediaPipe, bukan OpenCV
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Konversi warna (CPU-only karena OpenCV CUDA tidak tersedia)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe akan menggunakan GPU melalui TensorFlow Lite jika tersedia
        results = pose.process(image)
        
        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            keypoints_all.append(keypoints)
        else:
            # Jika tidak terdeteksi, tambahkan nol untuk menjaga keselarasan frame
            keypoints_all.append([0.0] * (33 * 3))  # 33 landmark dengan x,y,z
        
        # Progress reporting
        if frame_count % 100 == 0 or frame_count == total_frames:
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed if elapsed > 0 else 0
            remaining = (total_frames - frame_count) / fps_processing if fps_processing > 0 else 0
            print(f"Progress: {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%) | "
                  f"Processing speed: {fps_processing:.1f} FPS | "
                  f"Est. remaining: {remaining:.1f} seconds")
    
    elapsed_total = time.time() - start_time
    print(f"Total waktu pemrosesan: {elapsed_total:.2f} detik untuk {frame_count} frame "
          f"({frame_count/elapsed_total:.1f} FPS)")
    
    cap.release()
    return np.array(keypoints_all)

def process_all_videos():
    start_time_all = time.time()
    total_videos = 0
    processed_videos = 0
    
    for dance_name in os.listdir(RAW_VIDEO_BASE_DIR):
        dance_path = os.path.join(RAW_VIDEO_BASE_DIR, dance_name)
        output_path = os.path.join(OUTPUT_BASE_DIR, dance_name)
        
        if not os.path.isdir(dance_path):
            continue
        
        os.makedirs(output_path, exist_ok=True)
        
        # Hitung total video
        videos = [f for f in os.listdir(dance_path) if f.endswith('.mp4')]
        total_videos += len(videos)
        
        print(f"[{dance_name}] Menemukan {len(videos)} video untuk diproses")
        
        for filename in videos:
            video_path = os.path.join(dance_path, filename)
            npy_path = os.path.join(output_path, filename.replace('.mp4', '.npy'))
            
            if os.path.exists(npy_path):
                print(f"[{dance_name}] Lewati {filename}, sudah diproses sebelumnya.")
                processed_videos += 1
                continue
            
            print(f"\n[{dance_name}] Memproses {filename}...")
            keypoints = extract_pose_from_video(video_path)
            
            np.save(npy_path, keypoints)
            print(f"[{dance_name}] Disimpan ke {npy_path}, bentuk: {keypoints.shape}")
            processed_videos += 1
            
            # Status keseluruhan
            elapsed = time.time() - start_time_all
            print(f"Progress keseluruhan: {processed_videos}/{total_videos} video "
                  f"({processed_videos/total_videos*100:.1f}%)")
            
            # Bersihkan memori
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_time = time.time() - start_time_all
    print(f"\nSelesai! Total waktu: {total_time:.2f} detik untuk {processed_videos} video")

def print_mediapipe_info():
    """Try to get MediaPipe GPU info"""
    try:
        # Mencoba mendapatkan info tentang MediaPipe GPU
        from mediapipe.python._framework_bindings import resource_util
        gpu_resources = resource_util.list_gpu_devices()
        print(f"MediaPipe GPU resources: {gpu_resources}")
    except:
        print("Tidak dapat mendapatkan info MediaPipe GPU")

if __name__ == "__main__":
    # Informasi sistem
    print("\n===== INFORMASI SISTEM =====")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"MediaPipe version: {mp.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
    print("===========================\n")
    
    # Coba dapatkan info MediaPipe GPU
    print_mediapipe_info()
    
    # Periksa variabel lingkungan CUDA
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    process_all_videos()