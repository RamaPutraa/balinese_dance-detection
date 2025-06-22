import os
import numpy as np
import collections
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ========== Load Dataset ==========
X_train = np.load('data/result_processing/X_train.npy')
y_train = np.load('data/result_processing/y_train.npy')
X_val = np.load('data/result_processing/X_val.npy')
y_val = np.load('data/result_processing/y_val.npy')
X_test = np.load('data/result_processing/X_test.npy')
y_test = np.load('data/result_processing/y_test.npy')

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

LABELS = ['baris', 'gopala', 'pendet', 'puspanjali', 'sekar_jagat']
sequence_length = X_train.shape[1]   # 50
num_features = X_train.shape[2]      # 99
num_classes = y_train.shape[1]       # 5

# ========== Build Model ==========
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(sequence_length, num_features)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ========== Setup Checkpoint & EarlyStopping ==========
os.makedirs('src/model', exist_ok=True)
checkpoint = ModelCheckpoint('src/model/best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(patience=10, restore_best_weights=True)

# ========== Train Model ==========
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stop],
    verbose=1
)

print("📦 Training selesai, model disimpan di: src/model/best_model.h5")

# ========== Evaluasi dengan Test Set ==========
print("🧪 Evaluasi model dengan data test...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Final Test Accuracy: {acc:.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=LABELS))
print("\nDistribusi prediksi:", collections.Counter(y_pred_classes))

# ========== Confusion Matrix Heatmap ==========
cm = confusion_matrix(y_true_classes, y_pred_classes)
df_cm = pd.DataFrame(cm, index=LABELS, columns=LABELS)
plt.figure(figsize=(6,5))
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

# ========== Plot Akurasi dan Loss ==========
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
