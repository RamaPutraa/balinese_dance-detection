import os
import random
import numpy as np
import collections
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional,BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# ========== SET RANDOM SEED ==========
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# ========== Load Dataset ==========
X_train = np.load('data/result_processing_aug/X_train.npy')
y_train = np.load('data/result_processing_aug/y_train.npy')
X_val = np.load('data/result_processing_aug/X_val.npy')
y_val = np.load('data/result_processing_aug/y_val.npy')
X_test = np.load('data/result_processing_aug/X_test.npy')
y_test = np.load('data/result_processing_aug/y_test.npy')

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

LABELS = ['baris', 'gopala', 'pendet', 'puspanjali', 'sekar_jagat']
sequence_length = X_train.shape[1]
num_features = X_train.shape[2]
num_classes = y_train.shape[1]

# ========== Build Model ==========
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(50, 99)),
    Dropout(0.3),
    BatchNormalization(),

    Bidirectional(LSTM(64)),
    Dropout(0.3),
    BatchNormalization(),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(5, activation='softmax')  # 5 kelas
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ========== Setup Checkpoint & EarlyStopping ==========
os.makedirs('src/model', exist_ok=True)
checkpoint = ModelCheckpoint(
    'src/model/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    patience=15,
    restore_best_weights=True,
    monitor='val_loss'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# ========== Train Model ==========
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stop, reduce_lr],
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
