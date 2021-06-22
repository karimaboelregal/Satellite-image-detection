import numpy as np
import cv2
import matplotlib.pyplot as plt


image = cv2.pow((cv2.imread('dataset/4.jpg')/255.0), 1)

pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
print(pixel_values.shape)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 5
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
labels = labels.flatten()
masked_image = np.copy(image)

# convert to the shape of a vector of pixel values
masked_image = masked_image.reshape((-1, 3))

for i in range(5):
    highest = masked_image[labels == i][0]
    masked_image[labels == i] = highest



masked_image = masked_image.reshape(image.shape)

f, axarr = plt.subplots(1,3)

axarr[0].imshow(image)
axarr[1].imshow(masked_image)
axarr[2].imshow(masked_image)

plt.show()