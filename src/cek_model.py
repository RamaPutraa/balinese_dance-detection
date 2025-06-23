import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# === Load Model dan Data ===
MODEL_PATH = 'src/model/best_model.h5'
LABELS = ['Tari Baris', 'Tari Gopala', 'Tari Pendet', 'Tari Puspanjali', 'Tari Sekar Jagat']

model = load_model(MODEL_PATH)

X_test = np.load('data/result_processing_aug/X_test.npy')
y_test = np.load('data/result_processing_aug/y_test.npy')

print("Test set:", X_test.shape, y_test.shape)

# === Prediksi ===
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# === Evaluasi Umum ===
acc = np.mean(y_pred_classes == y_true_classes)
print(f"\n✅ Akurasi Test Set: {acc*100:.2f}%")

# === Classification Report ===
report = classification_report(y_true_classes, y_pred_classes, target_names=LABELS)
print("\n📄 Classification Report:")
print(report)

# === Confusion Matrix ===
cm = confusion_matrix(y_true_classes, y_pred_classes)
df_cm = pd.DataFrame(cm, index=LABELS, columns=LABELS)

plt.figure(figsize=(6,5))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# === Confidence Histogram ===
max_probs = np.max(y_pred_probs, axis=1)
plt.figure(figsize=(6,4))
plt.hist(max_probs, bins=20, color='skyblue', edgecolor='black')
plt.title('Confidence Distribution (Softmax Max)')
plt.xlabel('Confidence')
plt.ylabel('Jumlah Prediksi')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Prediksi Salah ===
wrong_indices = np.where(y_pred_classes != y_true_classes)[0]
print(f"\n🔍 Jumlah prediksi salah: {len(wrong_indices)} dari {len(y_test)}")

for i in wrong_indices[:10]:  # tampilkan 10 kesalahan pertama
    true_label = LABELS[y_true_classes[i]]
    pred_label = LABELS[y_pred_classes[i]]
    confidence = y_pred_probs[i][y_pred_classes[i]]
    print(f"Index {i}: Predicted = {pred_label} ({confidence:.2f}), Actual = {true_label}")

# === Distribusi Kelas Hasil Prediksi ===
print("\n📊 Distribusi Prediksi:")
counts = collections.Counter(y_pred_classes)
for idx, count in counts.items():
    print(f"{LABELS[idx]}: {count} samples")

