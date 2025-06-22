import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os

# =================
# Konfigurasi
# =================
DATA_PATH = 'data/result_processing'
NUM_CLASSES = 5
SEQUENCE_LENGTH = 50
NUM_FEATURES = 99
NUM_FOLDS = 5
LABELS = ['baris', 'gopala', 'pendet', 'puspanjali', 'sekar_jagat']

# =================
# Fungsi Model
# =================
def build_lstm_model():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(50, 99)),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# =================
# Load Dataset
# =================
X = np.load(os.path.join(DATA_PATH, 'X.npy'))
y = np.load(os.path.join(DATA_PATH, 'y.npy'))
y_cat = to_categorical(y, num_classes=NUM_CLASSES)

# =================
# K-Fold CV
# =================
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
all_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n==============================")
    print(f"🔁 Fold {fold+1}/{NUM_FOLDS}")
    print("==============================")

    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_cat[train_idx], y_cat[val_idx]

    # Bangun model
    model = build_lstm_model()

    # Callback early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Training
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=8,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=0
    )

    # Evaluasi
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    acc = accuracy_score(y_true_classes, y_pred_classes)
    all_accuracies.append(acc)
    print(f"✅ Accuracy Fold {fold+1}: {acc:.4f}")
    print("📊 Confusion Matrix:")
    print(confusion_matrix(y_true_classes, y_pred_classes))
    print("📋 Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=LABELS))

    # (Opsional) Simpan model
    model.save(f"src/model/fold_{fold+1}_model.h5")

# =================
# Final Result
# =================
mean_acc = np.mean(all_accuracies)
std_acc = np.std(all_accuracies)

print("\n==============================")
print("📈 K-Fold Evaluation Complete")
print("==============================")
print(f"🎯 Mean Accuracy: {mean_acc:.4f}")
print(f"📉 Std Dev:       {std_acc:.4f}")
