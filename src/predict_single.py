import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog

# Konfigurasi
LABELS = ['baris', 'pendet', 'rejang_sari']  # Sesuai urutan waktu training
MODEL_PATH = 'src/model/best_model.h5'       # Lokasi model yang sudah dilatih

# Load model
model = load_model(MODEL_PATH)

# Fungsi untuk prediksi 1 file .npy
def predict_dance(npy_path):
    data = np.load(npy_path)

    # Pastikan dimensinya sesuai: (sequence_length, num_features)
    if data.shape[0] != 50:  # 50 = MAX_FRAMES
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        data = pad_sequences([data], maxlen=50, dtype='float32', padding='post', truncating='post')
        data = data[0]

    data = np.expand_dims(data, axis=0)  # Jadi (1, 50, 99)

    # Prediksi
    prediction = model.predict(data)
    predicted_class = np.argmax(prediction)

    print(f"Prediksi: {LABELS[predicted_class]} (confidence: {np.max(prediction)*100:.2f}%)")

if __name__ == "__main__":
    # Buka jendela file picker
    root = Tk()
    root.withdraw()  # Supaya tidak muncul window Tkinter
    npy_file = filedialog.askopenfilename(title="Pilih file .npy untuk diprediksi",
                                          filetypes=[("NumPy Files", "*.npy")])

    if npy_file:
        predict_dance(npy_file)
    else:
        print("Tidak ada file yang dipilih.")
