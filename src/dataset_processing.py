import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# =================
# Konfigurasi
# =================
DATA_DIR = 'data/extracted_keypoints'
SAVE_DIR = 'data/result_processing'
MAX_FRAMES = 50
NUM_FEATURES = 99
LABELS = ['baris', 'gopala', 'pendet', 'puspanjali', 'sekar_jagat']
USE_AUGMENTATION = True

# =================
# Fungsi
# =================
def analyze_frame_distribution():
    lengths = []
    for dance_name in LABELS:
        folder = os.path.join(DATA_DIR, dance_name)
        for file in os.listdir(folder):
            if file.endswith('.npy'):
                data = np.load(os.path.join(folder, file))
                lengths.append(data.shape[0])
    plt.hist(lengths, bins=10)
    plt.title('Distribusi Panjang Frame')
    plt.xlabel('Jumlah Frame')
    plt.ylabel('Jumlah Video')
    plt.grid()
    plt.show()

def augment_keypoints(data):
    noise = np.random.normal(0, 0.01, data.shape)
    return data + noise

def load_keypoints():
    X = []
    y = []

    for label_idx, dance_name in enumerate(LABELS):
        dance_folder = os.path.join(DATA_DIR, dance_name)
        for file in os.listdir(dance_folder):
            if file.endswith('.npy'):
                data = np.load(os.path.join(dance_folder, file))

                # Normalisasi jumlah frame
                if data.shape[0] > MAX_FRAMES:
                    data = data[:MAX_FRAMES]
                elif data.shape[0] < MAX_FRAMES:
                    padding = np.zeros((MAX_FRAMES - data.shape[0], NUM_FEATURES))
                    data = np.vstack((data, padding))

                X.append(data)
                y.append(label_idx)

                # Augmentasi (opsional)
                if USE_AUGMENTATION:
                    X.append(augment_keypoints(data))
                    y.append(label_idx)

    return np.array(X), np.array(y)

def save_split(X_train, X_val, X_test, y_train, y_val, y_test, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
    print(f"💾 Dataset split disimpan di folder {save_dir}!")

# =================
# Main Flow
# =================
if __name__ == "__main__":
    print("📊 Menganalisis distribusi frame...")
    analyze_frame_distribution()

    print("🔁 Loading dan augmenting dataset...")
    X, y = load_keypoints()

    print("🔢 Melakukan one-hot encoding label...")
    y_cat = to_categorical(y, num_classes=len(LABELS))

    # Split Train-Test (80%) + Test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=np.argmax(y_cat, axis=1)
    )

    # Split Train-Validation (60% / 20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=np.argmax(y_temp, axis=1)
    )

    print("✅ Dataset siap digunakan!")
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)

    save_split(X_train, X_val, X_test, y_train, y_val, y_test, SAVE_DIR)
