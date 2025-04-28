import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Load dataset
X_train = np.load('data/result_processing/X_train.npy')
y_train = np.load('data/result_processing/y_train.npy')
X_test = np.load('data/result_processing/X_test.npy')
y_test = np.load('data/result_processing/y_test.npy')

# Cek shape dataset
print("X_train shape:", X_train.shape)  # (jumlah_data, sequence_length, num_features)
print("y_train shape:", y_train.shape)

# Parameter
sequence_length = X_train.shape[1]  # Harusnya 50
num_features = X_train.shape[2]     # Harusnya 99
num_classes = y_train.shape[1]       # Jumlah tarian (contoh 3)

# Build model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, num_features)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Save model terbaik otomatis
checkpoint = ModelCheckpoint('src/model/best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Training
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]
)

print("Training selesai, model disimpan di src/model/best_model.h5")
