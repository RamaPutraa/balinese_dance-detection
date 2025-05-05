import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
import time
import os
import queue
from concurrent.futures import ThreadPoolExecutor

class DanceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Pengenal Tarian Bali")
        self.root.geometry("900x650")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(True, True)

        # Konfigurasi
        self.MAX_FRAMES = 50
        self.NUM_FEATURES = 99
        self.LABELS = ['Tari Baris', 'Tari Pendet', 'Tari Rejang Sari']
        self.MODEL_PATH = 'src/model/best_model.h5'
        self.BATCH_SIZE = 4  # Ukuran batch untuk pemrosesan paralel
        
        # Variable
        self.video_path = None
        self.playing = False
        self.cap = None
        self.prediction_result = ""
        self.confidence = 0.0
        self.frame_with_pose = None
        self.frame_queue = queue.Queue(maxsize=30)  # Queue untuk pemrosesan frame
        self.result_queue = queue.Queue()  # Queue untuk hasil
        self.thread_pool = ThreadPoolExecutor(max_workers=4)  # Thread pool untuk pemrosesan paralel
        
        # Konfigurasi GPU
        self.setup_gpu()
        
        # Load model
        try:
            self.model = load_model(self.MODEL_PATH)
            # Lakukan warm-up untuk model
            dummy_input = np.zeros((1, self.MAX_FRAMES, self.NUM_FEATURES), dtype=np.float32)
            self.model.predict(dummy_input)
            print("Model berhasil dimuat dan warm-up selesai")
        except Exception as e:
            print(f"Error memuat model: {e}")
            self.model = None
        
        # Setup Mediapipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(static_image_mode=False, 
                                   min_detection_confidence=0.5, 
                                   min_tracking_confidence=0.5,
                                   model_complexity=1)  # Sesuaikan dengan kebutuhan (0=cepat, 1=seimbang, 2=akurat)
        
        # UI Setup
        self.create_widgets()
    
    def setup_gpu(self):
        """Konfigurasi penggunaan GPU"""
        try:
            # Cek ketersediaan GPU
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Konfigurasi untuk menggunakan GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Atur memori GPU secara dinamis berdasarkan kebutuhan
                gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
                tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
                
                print(f"GPU ditemukan: {gpus}")
                self.using_gpu = True
            else:
                print("GPU tidak ditemukan, menggunakan CPU")
                self.using_gpu = False
        except Exception as e:
            print(f"Error saat konfigurasi GPU: {e}")
            self.using_gpu = False
        
    def create_widgets(self):
        # Header frame
        header_frame = tk.Frame(self.root, bg="#2c3e50", pady=10)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(header_frame, text="SISTEM PENGENAL TARIAN BALI", 
                              font=("Arial", 18, "bold"), bg="#2c3e50", fg="white")
        title_label.pack(pady=5)
        
        # Tampilkan status GPU/CPU
        gpu_status = "GPU Aktif: RTX 3060" if self.using_gpu else "Mode CPU (GPU tidak terdeteksi)"
        gpu_label = tk.Label(header_frame, text=gpu_status,
                           font=("Arial", 10), bg="#2c3e50", fg="#72c5f3")
        gpu_label.pack(pady=2)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left frame (video display)
        left_frame = tk.Frame(main_frame, bg="#f0f0f0")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video display
        self.video_frame = tk.Frame(left_frame, bg="black", width=640, height=480)
        self.video_frame.pack(pady=10)
        self.video_frame.pack_propagate(False)
        
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Controls frame
        controls_frame = tk.Frame(left_frame, bg="#f0f0f0", pady=10)
        controls_frame.pack(fill=tk.X)
        
        # Button to select video
        self.select_btn = ttk.Button(
            controls_frame, 
            text="Pilih Video", 
            command=self.choose_file,
            style="TButton"
        )
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        # Button to start analysis
        self.analyze_btn = ttk.Button(
            controls_frame, 
            text="Analisis Video", 
            command=self.start_prediction,
            state=tk.DISABLED
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Speed settings frame
        speed_frame = tk.Frame(controls_frame, bg="#f0f0f0")
        speed_frame.pack(side=tk.RIGHT, padx=10)
        
        tk.Label(speed_frame, text="Kecepatan Deteksi:", bg="#f0f0f0").pack(side=tk.LEFT)
        
        self.speed_var = tk.StringVar(value="normal")
        speeds = [("Cepat", "fast"), ("Normal", "normal"), ("Akurat", "accurate")]
        
        for text, value in speeds:
            rb = ttk.Radiobutton(speed_frame, text=text, value=value, variable=self.speed_var)
            rb.pack(side=tk.LEFT, padx=5)
        
        # Right frame (results)
        right_frame = tk.Frame(main_frame, bg="#f0f0f0", width=250)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        # Results section
        results_frame = tk.LabelFrame(right_frame, text="Hasil Deteksi", bg="#f0f0f0", font=("Arial", 12, "bold"))
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # File info
        file_frame = tk.Frame(results_frame, bg="#f0f0f0", pady=5)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(file_frame, text="File Video:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.file_label = tk.Label(file_frame, text="Tidak ada file dipilih", bg="#f0f0f0", wraplength=230)
        self.file_label.pack(anchor=tk.W)
        
        # Separator
        ttk.Separator(results_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)
        
        # Prediction results
        prediction_frame = tk.Frame(results_frame, bg="#f0f0f0", pady=5)
        prediction_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(prediction_frame, text="Jenis Tarian:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.prediction_label = tk.Label(prediction_frame, text="-", bg="#f0f0f0", font=("Arial", 14))
        self.prediction_label.pack(anchor=tk.W, pady=5)
        
        tk.Label(prediction_frame, text="Confidence:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.confidence_frame = tk.Frame(prediction_frame, bg="#f0f0f0")
        self.confidence_frame.pack(fill=tk.X, pady=5)
        
        self.confidence_label = tk.Label(self.confidence_frame, text="0%", bg="#f0f0f0")
        self.confidence_label.pack(side=tk.RIGHT)
        
        self.confidence_bar = ttk.Progressbar(self.confidence_frame, orient="horizontal", length=200, mode="determinate")
        self.confidence_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Siap", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Apply custom style
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 10))
        
    def choose_file(self):
        """Opens file dialog to select video file"""
        video_path = filedialog.askopenfilename(
            title="Pilih Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        
        if video_path:
            self.video_path = video_path
            self.file_label.config(text=os.path.basename(video_path))
            self.analyze_btn.config(state=tk.NORMAL)
            self.reset_results()
            self.status_bar.config(text=f"Video dipilih: {os.path.basename(video_path)}")
            self.preview_video()
    
    def reset_results(self):
        """Reset result display"""
        self.prediction_label.config(text="-")
        self.confidence_label.config(text="0%")
        self.confidence_bar["value"] = 0
    
    def preview_video(self):
        """Show preview of selected video"""
        if self.playing:
            if self.cap:
                self.cap.release()
            self.playing = False
            return
        
        if not self.video_path:
            return
        
        # Reset video display
        self.playing = True
        
        def play_video():
            self.cap = cv2.VideoCapture(self.video_path)
            frame_count = 0
            
            while self.cap.isOpened() and self.playing:
                ret, frame = self.cap.read()
                if not ret:
                    # Loop video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Resize for display
                frame = cv2.resize(frame, (640, 480))
                
                # Convert to RGB for display
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
                
                # Control playback speed
                time.sleep(0.03)
                
                # Break if window is closed
                if not self.playing:
                    break
                    
            if self.cap:
                self.cap.release()
        
        # Start playback in a thread
        threading.Thread(target=play_video, daemon=True).start()
    
    def process_frame(self, frame):
        """Proses single frame untuk ekstraksi keypoints"""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize untuk membantu mempercepat deteksi
        h, w, _ = image.shape
        scale = min(640 / w, 480 / h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        # Deteksi pose
        results = self.pose.process(image)
        
        keypoints = None
        annotated_image = None
        
        if results.pose_landmarks:
            # Buat annotated image untuk visualisasi
            annotated_image = frame.copy()
            self.mp_drawing.draw_landmarks(
                annotated_image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # Ekstrak keypoints
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        
        return keypoints, annotated_image

    def extract_keypoints_from_video(self, video_path):
        """Extract pose keypoints from video dengan proses paralel"""
        cap = cv2.VideoCapture(video_path)
        keypoints_all = []
        
        # Update status
        self.status_bar.config(text="Mengekstrak fitur pose...")
        self.root.update()
        
        # Praproses video untuk mendapatkan semua frame
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        # Sesuaikan jika frame terlalu banyak (ambil sampel)
        total_frames = len(frames)
        if total_frames > 300:  # Batas atas jumlah frame yang akan diproses
            step = total_frames // 300
            frames = frames[::step][:300]
        
        # Proses batch frame secara paralel
        processed_frames = 0
        total_to_process = len(frames)
        display_interval = max(1, total_to_process // 10)  # Update tampilan setiap 10% progress
        
        # Bagi ke dalam batch untuk pemrosesan paralel
        for i in range(0, len(frames), self.BATCH_SIZE):
            batch = frames[i:i+self.BATCH_SIZE]
            futures = []
            
            # Submit batch untuk pemrosesan
            for frame in batch:
                futures.append(self.thread_pool.submit(self.process_frame, frame))
            
            # Tunggu hasil dan perbarui UI
            for future in futures:
                keypoints, annotated_image = future.result()
                processed_frames += 1
                
                if keypoints:
                    keypoints_all.append(keypoints)
                
                # Update preview dan progress setiap interval
                if annotated_image is not None and processed_frames % display_interval == 0:
                    # Resize dan tampilkan
                    annotated_image = cv2.resize(annotated_image, (640, 480))
                    cv2image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv2image)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    self.video_label.imgtk = imgtk
                    self.video_label.config(image=imgtk)
                    
                    # Update progress
                    progress = processed_frames / total_to_process * 100
                    self.status_bar.config(text=f"Mengekstrak fitur pose... {progress:.1f}%")
                    self.root.update()
        
        if not keypoints_all:
            return None
            
        keypoints_all = np.array(keypoints_all, dtype=np.float32)
        
        # Normalize frame count
        if keypoints_all.shape[0] > self.MAX_FRAMES:
            # Gunakan sampel dengan interval yang sama untuk mendapatkan MAX_FRAMES
            indices = np.linspace(0, keypoints_all.shape[0]-1, self.MAX_FRAMES, dtype=int)
            keypoints_all = keypoints_all[indices]
        elif keypoints_all.shape[0] < self.MAX_FRAMES:
            keypoints_all = pad_sequences([keypoints_all], maxlen=self.MAX_FRAMES, 
                                     dtype='float32', padding='post', truncating='post')[0]
            
        return keypoints_all
    
    def start_prediction(self):
        """Start prediction process in background thread dengan pengaturan kecepatan"""
        if not self.model:
            self.status_bar.config(text="Error: Model tidak ditemukan!")
            return
            
        if not self.video_path:
            self.status_bar.config(text="Error: Tidak ada video yang dipilih!")
            return
        
        # Atur konfigurasi berdasarkan pilihan kecepatan
        speed_mode = self.speed_var.get()
        if speed_mode == "fast":
            # Mode cepat: kurangi kompleksitas model dan ukuran batch besar
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
                model_complexity=0
            )
            self.BATCH_SIZE = 8
        elif speed_mode == "accurate":
            # Mode akurat: tingkatkan kompleksitas dan ukuran batch kecil
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
                model_complexity=2
            )
            self.BATCH_SIZE = 2
        else:  # normal
            # Mode seimbang
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            )
            self.BATCH_SIZE = 4
            
        # Disable buttons during processing
        self.select_btn.config(state=tk.DISABLED)
        self.analyze_btn.config(state=tk.DISABLED)
        self.reset_results()
        
        # Stop preview
        if self.playing:
            self.playing = False
            if self.cap:
                self.cap.release()
        
        # Run prediction in thread
        threading.Thread(target=self.predict_from_video, daemon=True).start()
    
    def predict_from_video(self):
        """Predict dance type from video dengan optimasi GPU"""
        try:
            # Extract keypoints
            keypoints = self.extract_keypoints_from_video(self.video_path)
            
            if keypoints is None:
                self.status_bar.config(text="Error: Tidak dapat mendeteksi pose dalam video!")
                self.select_btn.config(state=tk.NORMAL)
                self.analyze_btn.config(state=tk.NORMAL)
                return
                
            # Predict dengan GPU
            self.status_bar.config(text="Memprediksi jenis tarian...")
            self.root.update()
            
            # Konversi keypoints ke tensor dan format yang tepat
            keypoints = np.expand_dims(keypoints, axis=0).astype(np.float32)  # Shape (1, MAX_FRAMES, NUM_FEATURES)
            
            # Gunakan eager execution untuk prediksi cepat
            with tf.device('/GPU:0' if self.using_gpu else '/CPU:0'):
                prediction = self.model.predict(keypoints, verbose=0)
            
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            # Update UI
            self.prediction_label.config(text=self.LABELS[predicted_class])
            self.confidence_label.config(text=f"{confidence:.2f}%")
            self.confidence_bar["value"] = confidence
            
            # Color based on confidence
            if confidence > 80:
                self.prediction_label.config(fg="green")
            elif confidence > 50:
                self.prediction_label.config(fg="orange")
            else:
                self.prediction_label.config(fg="red")
                
            gpu_info = "dengan GPU" if self.using_gpu else "dengan CPU"
            self.status_bar.config(text=f"Prediksi selesai {gpu_info}: {self.LABELS[predicted_class]} ({confidence:.2f}%)")
            
        except Exception as e:
            self.status_bar.config(text=f"Error: {str(e)}")
            print(f"Error prediction: {e}")
            
        finally:
            # Re-enable buttons
            self.select_btn.config(state=tk.NORMAL)
            self.analyze_btn.config(state=tk.NORMAL)
            
            # Resume preview
            self.preview_video()

if __name__ == "__main__":
    root = tk.Tk()
    app = DanceRecognitionApp(root)
    root.mainloop()