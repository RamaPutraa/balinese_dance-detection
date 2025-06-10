import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
import time
import os
import glob

class DanceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Pengenal Tarian Bali")
        self.root.geometry("900x750")  # Sedikit diperbesar untuk area skor
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(True, True)

        # Konfigurasi
        self.MAX_FRAMES = 50
        self.NUM_FEATURES = 99
        self.LABELS = ['Tari Baris', 'Tari Pendet', 'Tari Rejang Sari']
        self.MODEL_PATH = 'src/model/best_model.h5'
        self.REFERENCE_PATH = 'data/extracted_keypoints'  # Path ke direktori data referensi
        
        # Variable
        self.video_path = None
        self.playing = False
        self.cap = None
        self.prediction_result = ""
        self.confidence = 0.0
        self.frame_with_pose = None
        self.extracted_keypoints = None  # Untuk menyimpan keypoints dari video yang dianalisis
        self.dance_score = 0.0  # Skor penilaian tarian
        
        # Load referensi keypoints
        self.reference_keypoints = self.load_reference_keypoints()
        
        # Load model
        try:
            self.model = load_model(self.MODEL_PATH)
            print("Model berhasil dimuat")
        except Exception as e:
            print(f"Error memuat model: {e}")
            self.model = None
        
        # Setup Mediapipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(static_image_mode=False, 
                                   min_detection_confidence=0.5, 
                                   min_tracking_confidence=0.5)
        
        # UI Setup
        self.create_widgets()
        
    def load_reference_keypoints(self):
        """Load semua file referensi keypoints"""
        reference_data = {}
        
        # Load untuk setiap jenis tarian
        for i, dance_type in enumerate(["baris", "pendet", "rejang_sari"]):
            reference_data[i] = []
            
            # Cari semua file .npy untuk tipe tarian ini
            pattern = os.path.join(self.REFERENCE_PATH, dance_type, "*.npy")
            files = glob.glob(pattern)
            
            for file in files:
                try:
                    keypoints = np.load(file)
                    reference_data[i].append(keypoints)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        
        return reference_data
        
    def create_widgets(self):
        # Header frame
        header_frame = tk.Frame(self.root, bg="#2c3e50", pady=10)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(header_frame, text="SISTEM PENGENAL TARIAN BALI", 
                              font=("Arial", 18, "bold"), bg="#2c3e50", fg="white")
        title_label.pack(pady=5)
        
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
        
        # Tambahkan bagian penilaian tarian
        ttk.Separator(results_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)
        
        score_frame = tk.Frame(results_frame, bg="#f0f0f0", pady=5)
        score_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(score_frame, text="Penilaian Tarian:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        # Frame untuk skor
        self.score_display_frame = tk.Frame(score_frame, bg="#f0f0f0")
        self.score_display_frame.pack(fill=tk.X, pady=5)
        
        # Gauge atau meter untuk skor
        self.score_gauge_frame = tk.Frame(score_frame, bg="#f0f0f0", height=30)
        self.score_gauge_frame.pack(fill=tk.X, pady=5)
        
        self.score_bar = ttk.Progressbar(self.score_gauge_frame, orient="horizontal", length=200, mode="determinate")
        self.score_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.score_label = tk.Label(self.score_gauge_frame, text="0/100", bg="#f0f0f0", font=("Arial", 14, "bold"))
        self.score_label.pack(side=tk.RIGHT)
        
        # Detail evaluasi
        self.evaluation_label = tk.Label(score_frame, text="-", bg="#f0f0f0", wraplength=230, justify=tk.LEFT)
        self.evaluation_label.pack(anchor=tk.W, pady=5)
        
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
        self.score_label.config(text="0/100")
        self.score_bar["value"] = 0
        self.evaluation_label.config(text="-")
    
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
    
    def extract_keypoints_from_video(self, video_path):
        """Extract pose keypoints from video"""
        cap = cv2.VideoCapture(video_path)
        keypoints_all = []
        
        # Update status
        self.status_bar.config(text="Mengekstrak fitur pose...")
        self.root.update()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
            
            # Show frame with pose landmarks
            if results.pose_landmarks:
                # Draw pose landmarks on image
                annotated_image = frame.copy()
                self.mp_drawing.draw_landmarks(
                    annotated_image, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS
                )
                
                # Update preview
                annotated_image = cv2.resize(annotated_image, (640, 480))
                cv2image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
                self.root.update()
                
                # Extract keypoints
                keypoints = []
                for lm in results.pose_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
                keypoints_all.append(keypoints)
                
            time.sleep(0.01)
            
        cap.release()
        
        if not keypoints_all:
            return None
            
        keypoints_all = np.array(keypoints_all)
        
        # Normalize frame count
        if keypoints_all.shape[0] > self.MAX_FRAMES:
            keypoints_all = keypoints_all[:self.MAX_FRAMES]
        elif keypoints_all.shape[0] < self.MAX_FRAMES:
            keypoints_all = pad_sequences([keypoints_all], maxlen=self.MAX_FRAMES, 
                                     dtype='float32', padding='post', truncating='post')[0]
            
        return keypoints_all
    
    def simple_dtw(self, s1, s2):
        """Implementasi sederhana dari Dynamic Time Warping"""
        n, m = len(s1), len(s2)
        dtw_matrix = np.zeros((n+1, m+1))
        
        for i in range(n+1):
            for j in range(m+1):
                dtw_matrix[i, j] = float('inf')
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = np.linalg.norm(s1[i-1] - s2[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                              dtw_matrix[i, j-1],    # deletion
                                              dtw_matrix[i-1, j-1])  # match
                
        # Return normalized distance
        return dtw_matrix[n, m] / (n + m)
        
    def calculate_dance_score(self, keypoints, predicted_class):
        """Calculate dance score by comparing with reference keypoints"""
        if predicted_class not in self.reference_keypoints or not self.reference_keypoints[predicted_class]:
            return 50.0, "Tidak dapat menilai: Data referensi tidak tersedia"
            
        # Get reference keypoints for the predicted dance
        references = self.reference_keypoints[predicted_class]
        
        if not references:
            return 50.0, "Tidak dapat menilai: Data referensi kosong"
            
        # Flatten keypoints for easier comparison (MAX_FRAMES x NUM_FEATURES)
        input_sequence = keypoints.reshape(-1, self.NUM_FEATURES)
        
        # Calculate similarity scores against all references
        similarity_scores = []
        dtw_scores = []
        
        for ref_keypoints in references:
            # Normalize frame count if needed
            if ref_keypoints.shape[0] > self.MAX_FRAMES:
                ref_keypoints = ref_keypoints[:self.MAX_FRAMES]
            elif ref_keypoints.shape[0] < self.MAX_FRAMES:
                ref_keypoints = pad_sequences([ref_keypoints], maxlen=self.MAX_FRAMES, 
                                         dtype='float32', padding='post', truncating='post')[0]
            
            ref_sequence = ref_keypoints.reshape(-1, self.NUM_FEATURES)
            
            try:
                # Metode 1: Perbandingan frame-by-frame
                frame_similarities = []
                
                for i in range(min(len(input_sequence), len(ref_sequence))):
                    # Hitung Euclidean distance
                    dist = np.linalg.norm(input_sequence[i] - ref_sequence[i])
                    # Konversi jarak ke similarity
                    similarity = np.exp(-dist / 10)
                    frame_similarities.append(similarity)
                
                avg_similarity = np.mean(frame_similarities) if frame_similarities else 0
                similarity_scores.append(avg_similarity)
                
                # Metode 2: Optional - DTW untuk gerakan-gerakan kunci (dioptimasi)
                # Pilih 15 keypoints penting saja untuk DTW (lebih efisien)
                # Misalnya: tangan, kaki, kepala, dan badan
                key_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # Indeks landmark penting
                
                # Ekstrak hanya landmark penting tiap frame
                input_key_seq = []
                ref_key_seq = []
                
                # Sampel hanya 10 frame untuk efisiensi (setiap n frame)
                step = max(1, len(input_sequence) // 10)
                
                for i in range(0, len(input_sequence), step):
                    if i < len(input_sequence):
                        # Ambil setiap 3 nilai (x,y,z) untuk keypoint yang dipilih
                        frame_data = []
                        for idx in key_indices:
                            pos = idx * 3
                            if pos+2 < len(input_sequence[i]):
                                frame_data.extend(input_sequence[i][pos:pos+3])
                        input_key_seq.append(np.array(frame_data))
                
                for i in range(0, len(ref_sequence), step):
                    if i < len(ref_sequence):
                        frame_data = []
                        for idx in key_indices:
                            pos = idx * 3
                            if pos+2 < len(ref_sequence[i]):
                                frame_data.extend(ref_sequence[i][pos:pos+3])
                        ref_key_seq.append(np.array(frame_data))
                
                # Hitung DTW jika data cukup
                if len(input_key_seq) > 0 and len(ref_key_seq) > 0:
                    # Hitung DTW
                    dtw_distance = self.simple_dtw(np.array(input_key_seq), np.array(ref_key_seq))
                    # Konversi ke similarity (0-1)
                    dtw_similarity = np.exp(-dtw_distance)
                    dtw_scores.append(dtw_similarity)
                
            except Exception as e:
                print(f"Error calculating similarity: {e}")
                similarity_scores.append(0)
                dtw_scores.append(0)
        
        # Get best scores
        best_sim_score = max(similarity_scores) if similarity_scores else 0
        best_dtw_score = max(dtw_scores) if dtw_scores else 0
        
        # Combine scores (70% frame-by-frame similarity, 30% DTW if available)
        if dtw_scores:
            combined_score = (best_sim_score * 0.7) + (best_dtw_score * 0.3)
        else:
            combined_score = best_sim_score
        
        # Convert to 0-100 scale
        score = min(100, max(0, combined_score * 100))
        
        # Generate evaluation text
        evaluation = self.generate_evaluation_text(score)
        
        return score, evaluation
        
    def generate_evaluation_text(self, score):
        """Generate evaluation text based on score"""
        if score >= 90:
            return "Sempurna! Gerakan tarian sangat akurat dan sesuai dengan standar."
        elif score >= 80:
            return "Sangat baik! Gerakan tarian hampir sempurna dengan sedikit penyesuaian."
        elif score >= 70:
            return "Baik. Gerakan tarian sesuai tetapi memerlukan beberapa perbaikan."
        elif score >= 60:
            return "Cukup baik. Gerakan dasar sudah benar namun perlu konsistensi."
        elif score >= 50:
            return "Sedang. Gerakan dasar terlihat tetapi perlu banyak latihan."
        elif score >= 30:
            return "Perlu latihan lebih. Beberapa gerakan tidak sesuai dengan standar."
        else:
            return "Perlu banyak latihan. Banyak gerakan yang tidak sesuai dengan standar."
    
    def start_prediction(self):
        """Start prediction process in background thread"""
        if not self.model:
            self.status_bar.config(text="Error: Model tidak ditemukan!")
            return
            
        if not self.video_path:
            self.status_bar.config(text="Error: Tidak ada video yang dipilih!")
            return
            
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
        """Predict dance type from video"""
        try:
            # Extract keypoints
            keypoints = self.extract_keypoints_from_video(self.video_path)
            self.extracted_keypoints = keypoints  # Simpan untuk perhitungan skor
            
            if keypoints is None:
                self.status_bar.config(text="Error: Tidak dapat mendeteksi pose dalam video!")
                self.select_btn.config(state=tk.NORMAL)
                self.analyze_btn.config(state=tk.NORMAL)
                return
                
            # Predict
            self.status_bar.config(text="Memprediksi jenis tarian...")
            self.root.update()
            
            keypoints_input = np.expand_dims(keypoints, axis=0)  # Shape (1, MAX_FRAMES, NUM_FEATURES)
            prediction = self.model.predict(keypoints_input)
            
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            # Update UI for prediction
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
            
            # Calculate dance score
            self.status_bar.config(text="Menghitung nilai tarian...")
            self.root.update()
            
            score, evaluation = self.calculate_dance_score(keypoints, predicted_class)
            
            # Update UI for score
            self.score_label.config(text=f"{score:.1f}/100")
            self.score_bar["value"] = score
            self.evaluation_label.config(text=evaluation)
            
            # Color based on score
            if score >= 80:
                self.score_label.config(fg="green")
            elif score >= 60:
                self.score_label.config(fg="orange")
            else:
                self.score_label.config(fg="red")
                
            self.status_bar.config(text=f"Analisis selesai: {self.LABELS[predicted_class]} ({confidence:.2f}%), Skor: {score:.1f}/100")
            
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