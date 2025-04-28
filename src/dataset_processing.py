import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# =================
# Konfigurasi
# =================
DATA_DIR = 'data/extracted_keypoints'
SAVE_DIR = 'data/result_processing'   # Folder untuk hasil split
MAX_FRAMES = 50  # Jumlah frame tetap (sequence length)
NUM_FEATURES = 99  # 33 keypoint * 3 koordinat
LABELS = ['baris', 'pendet', 'rejang_sari']  # urutan label

def load_keypoints():
    X = []
    y = []

    for label_idx, dance_name in enumerate(LABELS):
        dance_folder = os.path.join(DATA_DIR, dance_name)
        for file in os.listdir(dance_folder):
            if file.endswith('.npy'):
                data = np.load(os.path.join(dance_folder, file))

                # Normalisasi jumlah frame: potong atau padding
                if data.shape[0] > MAX_FRAMES:
                    data = data[:MAX_FRAMES]
                elif data.shape[0] < MAX_FRAMES:
                    padding = np.zeros((MAX_FRAMES - data.shape[0], NUM_FEATURES))
                    data = np.vstack((data, padding))

                X.append(data)
                y.append(label_idx)

    return np.array(X), np.array(y)

def save_split(X_train, X_test, y_train, y_test, save_dir):
    # Buat folder kalau belum ada
    os.makedirs(save_dir, exist_ok=True)

    # Simpan file
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)

    print(f"💾 Dataset split disimpan di folder {save_dir}!")


# =================
# Main Flow
# =================
if __name__ == "__main__":
    print("🔁 Loading and processing dataset...")
    X, y = load_keypoints()

    # One-hot encoding label
    y_cat = to_categorical(y, num_classes=len(LABELS))

    # Split data train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )

    print("✅ Dataset siap digunakan!")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # Simpan hasil split
    save_split(X_train, X_test, y_train, y_test, SAVE_DIR)
