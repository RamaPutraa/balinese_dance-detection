import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
import os
from scipy.spatial.distance import cosine
import glob


class DanceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Pengenal Tarian Bali")
        self.root.geometry("900x650")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(True, True)

        # Konfigurasi
        self.MAX_FRAMES = 50
        self.NUM_FEATURES = 99  # 33 titik x, y, z
        self.LABELS = ['Tari Baris', 'Tari Pendet', 'Tari Rejang Sari']
        self.MODEL_PATH = 'src/model/best_model.h5'
        self.REFERENCE_DIR = 'data/extracted_keypoints/'

        # Variabel state
        self.video_path = None
        self.playing = False
        self.cap = None
        self.extracted_keypoints = None

        # Load model
        try:
            self.model = load_model(self.MODEL_PATH)
            print("✅ Model berhasil dimuat")
        except Exception as e:
            print(f"❌ Gagal memuat model: {e}")
            self.model = None

        # Mediapipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

        # UI
        self.create_widgets()

    def create_widgets(self):
        # Header
        header = tk.Frame(self.root, bg="#2c3e50")
        header.pack(fill=tk.X)
        tk.Label(header, text="SISTEM PENGENAL TARIAN BALI", font=("Arial", 18, "bold"),
                 bg="#2c3e50", fg="white").pack(pady=10)

        main = tk.Frame(self.root, bg="#f0f0f0")
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        left = tk.Frame(main, bg="#f0f0f0")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_frame = tk.Frame(left, bg="black", width=640, height=480)
        self.video_frame.pack()
        self.video_frame.pack_propagate(False)
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        control = tk.Frame(left, bg="#f0f0f0")
        control.pack(pady=10)
        self.select_btn = ttk.Button(control, text="Pilih Video", command=self.choose_file)
        self.select_btn.pack(side=tk.LEFT, padx=5)
        self.analyze_btn = ttk.Button(control, text="Analisis Video", command=self.start_prediction, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        self.load_npy_btn = ttk.Button(control, text="Muat File NPY", command=self.load_npy_file)
        self.load_npy_btn.pack(side=tk.LEFT, padx=5)
        self.save_keypoints_btn = ttk.Button(control, text="Simpan Keypoints", command=self.save_keypoints, state=tk.DISABLED)
        self.save_keypoints_btn.pack(side=tk.LEFT, padx=5)

        right = tk.Frame(main, bg="#f0f0f0", width=280)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        result_frame = tk.LabelFrame(right, text="Hasil Deteksi", font=("Arial", 12, "bold"), bg="#f0f0f0")
        result_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(result_frame, text="File Video:", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(anchor=tk.W, padx=10)
        self.file_label = tk.Label(result_frame, text="Tidak ada file", bg="#f0f0f0", wraplength=250)
        self.file_label.pack(anchor=tk.W, padx=10)

        ttk.Separator(result_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        tk.Label(result_frame, text="Hasil Model:", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(anchor=tk.W, padx=10)
        self.prediction_label = tk.Label(result_frame, text="-", font=("Arial", 14), bg="#f0f0f0")
        self.prediction_label.pack(anchor=tk.W, padx=10, pady=5)

        tk.Label(result_frame, text="Confidence:", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(anchor=tk.W, padx=10)
        self.confidence_bar = ttk.Progressbar(result_frame, length=200, mode='determinate')
        self.confidence_bar.pack(padx=10, pady=(0, 5))
        self.confidence_label = tk.Label(result_frame, text="0%", bg="#f0f0f0")
        self.confidence_label.pack(anchor=tk.E, padx=10)

        ttk.Separator(result_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        tk.Label(result_frame, text="Kecocokan dengan Referensi:", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(anchor=tk.W, padx=10)
        self.similarity_frame = tk.Frame(result_frame, bg="#f0f0f0")
        self.similarity_frame.pack(fill=tk.X, padx=10, pady=5)

        self.similarity_labels = {}
        self.similarity_bars = {}
        self.similarity_percentages = {}

        for i, label in enumerate(self.LABELS):
            frame = tk.Frame(self.similarity_frame, bg="#f0f0f0")
            frame.pack(fill=tk.X, pady=4)
            tk.Label(frame, text=f"{label}:", bg="#f0f0f0", width=12, anchor="w").pack(side=tk.LEFT)
            self.similarity_bars[label] = ttk.Progressbar(frame, length=150, mode='determinate')
            self.similarity_bars[label].pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.similarity_percentages[label] = tk.Label(frame, text="0%", bg="#f0f0f0", width=8)
            self.similarity_percentages[label].pack(side=tk.RIGHT)

        self.detail_frame = tk.Frame(result_frame, bg="#f0f0f0")
        self.detail_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(self.detail_frame, text="File Referensi Terbaik:", font=("Arial", 9, "bold"), bg="#f0f0f0").pack(anchor=tk.W)
        self.best_match_label = tk.Label(self.detail_frame, text="-", bg="#f0f0f0", wraplength=250)
        self.best_match_label.pack(anchor=tk.W)

        self.status_bar = tk.Label(self.root, text="Memuat referensi...", bd=1, relief=tk.SUNKEN, anchor=tk.W,
                                   font=("Arial", 9))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        style = ttk.Style()
        style.configure("TButton", font=("Arial", 10))

    def choose_file(self):
        path = filedialog.askopenfilename(title="Pilih Video",
                                          filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if path:
            self.video_path = path
            self.file_label.config(text=os.path.basename(path))
            self.analyze_btn.config(state=tk.NORMAL)
            self.reset_results()
            self.status_bar.config(text="Video dipilih")
            self.preview_video()

    def reset_results(self):
        self.prediction_label.config(text="-", fg="black")
        self.confidence_label.config(text="0%")
        self.confidence_bar["value"] = 0
        self.save_keypoints_btn.config(state=tk.DISABLED)
        self.extracted_keypoints = None

        for label in self.LABELS:
            self.similarity_bars[label]["value"] = 0
            self.similarity_percentages[label].config(text="0%")

    def preview_video(self):
        if self.playing:
            self.playing = False
            if self.cap:
                self.cap.release()
            return
        self.playing = True

        def play():
            self.cap = cv2.VideoCapture(self.video_path)
            while self.cap.isOpened() and self.playing:
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frame = cv2.resize(frame, (640, 480))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.video_label.imgtk = img
                self.video_label.config(image=img)
                time.sleep(0.03)
            if self.cap:
                self.cap.release()

        threading.Thread(target=play, daemon=True).start()

    def extract_keypoints_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        keypoints_all = []
        pose_detected = False
        self.status_bar.config(text="🔍 Mengekstrak pose...")
        self.root.update()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
            if results.pose_landmarks:
                pose_detected = True
                keypoints = [coord for lm in results.pose_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
                keypoints_all.append(keypoints)
                annotated = frame.copy()
                self.mp_drawing.draw_landmarks(annotated, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                preview = cv2.cvtColor(cv2.resize(annotated, (640, 480)), cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(preview))
                self.video_label.imgtk = img
                self.video_label.config(image=img)
                self.root.update()
            time.sleep(0.01)

        cap.release()
        if not pose_detected:
            return None

        keypoints_all = np.array(keypoints_all)
        if len(keypoints_all) > self.MAX_FRAMES:
            keypoints_all = keypoints_all[:self.MAX_FRAMES]
        else:
            keypoints_all = pad_sequences([keypoints_all], maxlen=self.MAX_FRAMES, dtype='float32', padding='post')[0]
        return keypoints_all

    def start_prediction(self):
        if not self.model:
            self.status_bar.config(text="❌ Model tidak tersedia")
            return
        if not self.video_path:
            self.status_bar.config(text="❌ Video belum dipilih")
            return

        self.select_btn.config(state=tk.DISABLED)
        self.analyze_btn.config(state=tk.DISABLED)
        self.reset_results()

        if self.playing:
            self.playing = False
            if self.cap:
                self.cap.release()

        threading.Thread(target=self.predict_from_video, daemon=True).start()

    def predict_from_video(self):
        try:
            keypoints = self.extract_keypoints_from_video(self.video_path)
            if keypoints is None:
                self.status_bar.config(text="❌ Pose tidak terdeteksi pada video.")
                return

            self.extracted_keypoints = keypoints
            self.save_keypoints_btn.config(state=tk.NORMAL)
            self.status_bar.config(text="⏳ Memprediksi jenis tarian...")
            self.root.update()

            prediction = self.model.predict(np.expand_dims(keypoints, axis=0))
            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction)) * 100
            predicted_label = self.LABELS[predicted_class]

            self.prediction_label.config(text=predicted_label)
            self.confidence_label.config(text=f"{confidence:.2f}%")
            self.confidence_bar["value"] = confidence

            if confidence > 80:
                self.prediction_label.config(fg="green")
            elif confidence > 50:
                self.prediction_label.config(fg="orange")
            else:
                self.prediction_label.config(fg="red")

            self.calculate_similarity(keypoints, predicted_label)
            self.show_evaluation_result()
            self.status_bar.config(
                text=f"✅ Prediksi: {predicted_label} ({confidence:.2f}%)"
            )
        except Exception as e:
            self.status_bar.config(text=f"❌ Error prediksi: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.select_btn.config(state=tk.NORMAL)
            self.analyze_btn.config(state=tk.NORMAL)
            self.preview_video()

        def calculate_similarity(self, keypoints, predicted_label):
            dir_map = {
                'Tari Baris': 'baris',
                'Tari Pendet': 'pendet',
                'Tari Rejang Sari': 'rejang_sari'
            }
            
            if predicted_label not in dir_map:
                self.status_bar.config(text="⚠️ Label tidak valid")
                return

            dance_dir = os.path.join(self.REFERENCE_DIR, dir_map[predicted_label])
            if not os.path.exists(dance_dir):
                self.status_bar.config(text="⚠️ Direktori referensi tidak ditemukan")
                return

            ref_files = glob.glob(os.path.join(dance_dir, "*.npy"))
            if not ref_files:
                self.status_bar.config(text="⚠️ Tidak ada file referensi")
                return

            best_score = 0
            best_file = None

            for file in ref_files:
                try:
                    ref_kp = np.load(file)
                    if ref_kp.shape != (self.MAX_FRAMES, self.NUM_FEATURES):
                        continue

                    sim_scores = []
                    for i in range(self.MAX_FRAMES):
                        if np.all(keypoints[i] == 0) or np.all(ref_kp[i] == 0):
                            continue
                        sim = 1 - cosine(keypoints[i], ref_kp[i])
                        if not np.isnan(sim):
                            sim_scores.append(sim)

                    avg_sim = np.mean(sim_scores) if sim_scores else 0
                    if avg_sim > best_score:
                        best_score = avg_sim
                        best_file = file
                except Exception as e:
                    print(f"Error saat membaca {file}: {e}")

            match_percentage = max(0, min(100, best_score * 100))
            self.best_match_keypoints = np.load(best_file)  # Simpan keypoints terbaik
            self.similarity_score = match_percentage

            # Update UI
            self.similarity_bars[predicted_label]["value"] = match_percentage
            self.similarity_percentages[predicted_label].config(text=f"{match_percentage:.1f}%")
            self.best_match_label.config(text=f"{os.path.basename(best_file)} ({match_percentage:.1f}%)")
            self.status_bar.config(text=f"✅ Cocok dengan: {os.path.basename(best_file)} ({match_percentage:.1f}%)")

    def load_npy_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("NPY files", "*.npy")])
        if not file_path:
            return
        try:
            keypoints = np.load(file_path)
            if len(keypoints.shape) != 2 or keypoints.shape[1] != self.NUM_FEATURES:
                raise ValueError("Format NPY tidak valid")
            if keypoints.shape[0] != self.MAX_FRAMES:
                if keypoints.shape[0] > self.MAX_FRAMES:
                    keypoints = keypoints[:self.MAX_FRAMES]
                else:
                    keypoints = pad_sequences([keypoints], maxlen=self.MAX_FRAMES, dtype='float32', padding='post')[0]
            self.extracted_keypoints = keypoints
            self.save_keypoints_btn.config(state=tk.NORMAL)
            self.file_label.config(text=os.path.basename(file_path))
            self.status_bar.config(text="⏳ Memprediksi dari file NPY...")
            self.root.update()
            prediction = self.model.predict(np.expand_dims(keypoints, axis=0))
            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction)) * 100
            predicted_label = self.LABELS[predicted_class]
            self.prediction_label.config(text=predicted_label)
            self.confidence_label.config(text=f"{confidence:.2f}%")
            self.confidence_bar["value"] = confidence
            self.calculate_similarity(keypoints, predicted_label)
        except Exception as e:
            self.status_bar.config(text=f"❌ Gagal memuat file NPY: {e}")
            messagebox.showerror("Error", f"Gagal memuat file NPY: {str(e)}")

    def save_keypoints(self):
        if self.extracted_keypoints is None:
            messagebox.showinfo("Info", "Tidak ada keypoints untuk disimpan")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("NumPy files", "*.npy")])
        if file_path:
            try:
                np.save(file_path, self.extracted_keypoints)
                messagebox.showinfo("Sukses", f"Keypoints berhasil disimpan ke:\n{file_path}")
                self.status_bar.config(text=f"✅ Keypoints disimpan: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menyimpan file: {str(e)}")
                self.status_bar.config(text=f"❌ Gagal menyimpan: {str(e)}")

    def show_evaluation_result(self):
        if not hasattr(self, 'similarity_score'):
            messagebox.showwarning("Peringatan", "Belum ada hasil kesamaan yang tersedia.")
            return

        score = self.similarity_score
        result_text = f"Nilai Tarian: {score:.1f}%\n"
        if score >= 80:
            result_text += "Kategori: Sangat Baik"
        elif score >= 60:
            result_text += "Kategori: Cukup Baik"
        elif score >= 40:
            result_text += "Kategori: Perlu Latihan"
        else:
            result_text += "Kategori: Perlu Bimbingan"

        messagebox.showinfo("Hasil Evaluasi", result_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = DanceRecognitionApp(root)
    root.mainloop()