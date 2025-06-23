from tensorflow.keras.models import load_model
import numpy as np

model = load_model('src/model/best_model.h5')
LABELS = ['Tari Baris', 'Tari Gopala', 'Tari Pendet', 'Tari Puspanjali', 'Tari Sekar Jagat']

keypoints = np.load('data/extracted_keypoints/puspanjali/file1.npy')  # file hasil ekstraksi
keypoints = np.expand_dims(keypoints, axis=0)

pred = model.predict(keypoints)
pred_class = np.argmax(pred)
print("Predicted:", LABELS[pred_class])
print("Confidence:", pred[0][pred_class])
