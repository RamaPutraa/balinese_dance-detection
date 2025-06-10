import tensorflow as tf

# Muat model Keras
model = tf.keras.models.load_model("src/model/best_model.h5")

# Konversi ke TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantisasi untuk efisiensi
tflite_model = converter.convert()

# Simpan model TFLite
with open("src/model/dance_model.tflite", "wb") as f:
    f.write(tflite_model)