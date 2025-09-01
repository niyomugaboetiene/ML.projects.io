import numpy as np
# sklearn.datasets is library for ML used of experimenting image recognation
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

print ("Distance between tow 0's: ", dist_same)
print("Distance between '0' and '5'", dist_diff)

# show the images
fig, axes = plt.subplots(1, 3, figsize=(6, 2))
axes[0].imshow(img1, cmap='gray'); axes[0].set_title("Digit 0")
axes[1].imshow(img2, cmap='gray'); axes[1].set_title("Digit 1")
axes[2].imshow(img3, cmap="gray"); axes[2].set_title("Digit 5")

plt.show()