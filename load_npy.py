import numpy as np
from utils.visualization import visualize_keypoints

data = np.load('data/extracted_keypoints/pendet/file1.npy')
print(data.shape)
print(data[0])
    
visualize_keypoints("data/extracted_keypoints/pendet/file1.npy")