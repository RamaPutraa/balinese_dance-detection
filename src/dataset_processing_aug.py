import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import scipy.interpolate
import random
from collections import Counter
import tensorflow as tf
# =================
# Konfigurasi
# =================
DATA_DIR = 'data/extracted_keypoints'
SAVE_DIR = 'data/result_processing_aug'
MAX_FRAMES = 50
NUM_FEATURES = 99
LABELS = ['baris', 'gopala', 'pendet', 'puspanjali', 'sekar_jagat']
USE_AUGMENTATION = True
N_AUGMENT = 3 
# =================
# Fungsi
# =================

# ========== SET RANDOM SEED ==========
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# Jitter koordinat
def augment_jitter(data, sigma=0.02):
    noise = np.random.normal(loc=0, scale=sigma, size=data.shape)
    return data + noise

# Drop frame dan padding ulang
def augment_drop_frames(data, drop_rate=0.1):
    num_frames = data.shape[0]
    drop_count = int(num_frames * drop_rate)

    if drop_count >= num_frames - 1:
        return data  # hindari error

    keep_indices = sorted(np.random.choice(num_frames, num_frames - drop_count, replace=False))
    dropped = data[keep_indices]

    if dropped.shape[0] < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - dropped.shape[0], NUM_FEATURES))
        return np.vstack((dropped, pad))
    else:
        return dropped[:MAX_FRAMES]

# Time warp dengan interpolasi
def augment_time_warp(data, warp_factor=1.1):
    original_len = data.shape[0]
    x_old = np.linspace(0, 1, original_len)
    x_new = np.linspace(0, 1, MAX_FRAMES)

    warped = np.zeros((MAX_FRAMES, NUM_FEATURES))
    for i in range(NUM_FEATURES):
        f = scipy.interpolate.interp1d(x_old, data[:, i], kind='linear', fill_value="extrapolate")
        warped[:, i] = f(x_new)
    return warped

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

def load_keypoints():
    X = []
    y = []

    augment_fns = [augment_jitter, augment_drop_frames, augment_time_warp]

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

                # Data asli
                X.append(data)
                y.append(label_idx)

                # Data augmentasi
                if USE_AUGMENTATION:
                    for _ in range(N_AUGMENT):
                        fn = random.choice(augment_fns)
                        aug_data = fn(data)
                        X.append(aug_data)
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
    print("🔍 Distribusi label:", Counter(np.argmax(y_cat, axis=1)))
    print(f"📈 Total data setelah augmentasi: {len(X)} sample")

    save_split(X_train, X_val, X_test, y_train, y_val, y_test, SAVE_DIR)
