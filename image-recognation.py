import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# load digits
digits = load_digits()

# pick two images from the dataset
img1 = digits.images[0]
img2 = digits.images[1]
img3 = digits.images[10]

# flatten image into feature vectors (64 pixels)

vec1 = img1.flatten()
vec2 = img2.flatten()
vec3 = img3.flatten()

# compute Euclidean distances
dist_same = np.sqrt(np.sum((vec1 - vec2) ** 2))
dist_diff = np.sqrt(np.sum((vec1 - vec3) ** 2))

print ("")