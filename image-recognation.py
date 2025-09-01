import numpy as np
# sklearn.datasets is library for ML used of experimenting image recognation
from sklearn.datasets import load_digits
# load_digits is built-in dataset of handwritten digits from 0 to 9
import matplotlib.pyplot as plt
# displaying image

# load digits
# is used to load handwritten-digits dataset into memory
digits = load_digits()

# pick two images from the dataset
# select image 1
img1 = digits.images[0]
# select second image
img2 = digits.images[1]
# select eleven image
img3 = digits.images[10]

# flatten image into feature vectors (64 pixels)
# convert the 2D array of image into 1D for better mathematical caliculation
vec1 = img1.flatten()
vec2 = img2.flatten()
vec3 = img3.flatten()

# compute Euclidean distances
dist_same = np.sqrt(np.sum((vec1 - vec2) ** 2))
dist_diff = np.sqrt(np.sum((vec1 - vec3) ** 2))

print ("Distance between tow 0's: ", dist_same)
print("Distance between '0' and '5'", dist_diff)

# show the images
# create image with 1 row and 3 column, each image has 6 width and 2 height
# axes is the list of axes image 
# axes[0] is the lef, axes[1] is the middle image, axes[2] is the right image
# axes it like each single image
fig, axes = plt.subplots(1, 3, figsize=(6, 2))
# fig is list of figure image canvas
# cmap it specify color map, imshow display 2D array as iimage, set_title -> adds title to each image
axes[0].imshow(img1, cmap='gray'); axes[0].set_title("Digit 0")
axes[1].imshow(img2, cmap='gray'); axes[1].set_title("Digit 1")
axes[2].imshow(img3, cmap="gray"); axes[2].set_title("Digit 5")

plt.show()