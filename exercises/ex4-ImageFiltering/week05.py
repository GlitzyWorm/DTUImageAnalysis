import numpy as np
from scipy.ndimage import correlate
from skimage import io
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.filters import gaussian
from skimage.filters import prewitt_h
from skimage.filters import prewitt_v
from skimage.filters import prewitt
from skimage.filters import threshold_otsu

# input_img = np.arange(25).reshape(5, 5)
# print(input_img)
#
# weights = [[0, 1, 0],
#            [1, 2, 1],
#            [0, 1, 0]]
#
# res_img = correlate(input_img, weights)

# Exercise 1: Print the value in position (3, 3) in res_img. Explain the value?
# print(res_img[3, 3])

# Exercise 2: Compare the output images when using reflection and
# constant for the border. Where and why do you see the differences.
# res_img = correlate(input_img, weights, mode="constant", cval=10)
# print(res_img)
#
# res_img = correlate(input_img, weights, mode="reflect")
# print(res_img)

# Edges are different because of the way the border is handled.
# In the first case, the border is filled with a constant value, while in the second case, the border is reflected.

# Exercise 3: Mean filtering
# Read and show the image Gaussian.png
img = io.imread("data/Gaussian.png")
# io.imshow(img)
# io.show()

# Create a mean filter with normalized weights
# size = 5
# # Two-dimensional filter filled with 1
# weights = np.ones([size, size, 3])
# # Normalize weights
# weights = weights / np.sum(weights)

# Use correlate with the Gaussian.png image and the mean filter.
# Show the resulting image together with the input image. What do you observe?
# res_img = correlate(img, weights)
#
# fig, ax = plt.subplots(ncols=2)
# ax[0].imshow(img)
# ax[1].imshow(res_img)
# for a in ax:
#     a.axis('off')
# plt.tight_layout()
# plt.show()

# Exercise 4: Median filtering

# We can create a footprint which marks the size of the median filter and do the filtering like this:
# size = 30
# footprint = np.ones([size, size, 3])
# med_img = median(img, footprint)

# Filter the Gaussian.png image with the median filter with different size (5, 10, 20...).
# What do you observe? What happens with the noise and with the lighth-dark transitions?

# fig, ax = plt.subplots(ncols=2)
# ax[0].imshow(img)
# ax[1].imshow(med_img)
# for a in ax:
#     a.axis('off')
# plt.tight_layout()
# plt.show()

# Create a for loop that iterates over different sizes and shows the resulting images.
# sizes = [5, 10, 20, 30]
# for size in sizes:
#     footprint = np.ones([size, size, 3])
#     med_img = median(img, footprint)
#     plt.imshow(med_img)
#     plt.show()

# Exercise 5: Gaussian filtering
# sigma = 1
# gauss_img = gaussian(img, sigma)

# Try to change the sigma value and observe the result.
# sigmas = [1, 2, 3, 4]
# for sigma in sigmas:
#     gauss_img = gaussian(img, sigma, channel_axis=-1)
#     plt.imshow(gauss_img)
#     plt.show()

# Exercise 8 and 9: Prewitt filtering

# Load donald_1.png as grayscale and apply the prewitt filter to the image.
# img = io.imread("data/donald_1.png", as_gray=True)

# prewitt_img_h = prewitt_h(img)
# prewitt_img_v = prewitt_v(img)

# Print the values of the resulting images.
# print(prewitt_img_h[:, :])
# print(prewitt_img_v[:, :])

# Prewitt filter
# prewitt_img = prewitt(img)

# Show the resulting images together with the input image. What do you observe?
# Show as grayscale images.
# fig, ax = plt.subplots(ncols=4)
# ax[0].imshow(img, cmap="gray")
# ax[1].imshow(prewitt_img_h, cmap="gray")
# ax[2].imshow(prewitt_img_v, cmap="gray")
# ax[3].imshow(prewitt_img, cmap="gray")
# for a in ax:
#     a.axis('off')
# plt.tight_layout()
# plt.show()

# Exercise 10: Edge detection

# Load ElbowCTSlice.png as grayscale
img = io.imread("data/ElbowCTSlice.png", as_gray=True)

# Show the image
# io.imshow(img)
# io.show()

# Filter the image using either a Gaussian filter or a median filter

# 1. Gaussian filter
# sigma = 1
# gauss_img = gaussian(img, sigma)

# 2. Median filter
# size = 5
# footprint = np.ones([size, size])
# med_img = median(img, footprint)

# 3. Gaussian filter and Median filter
# sigma = 1
# gauss_img_2 = gaussian(img, sigma)
# size = 5
# footprint_2 = np.ones([size, size])
# gauss_med_img = median(gauss_img_2, footprint_2)

# 4. Median filter and Gaussian filter
sigma = 2
size = 10
footprint_3 = np.ones([size, size])
med_gauss_img = gaussian(median(img, footprint_3), sigma)

# Compute the gradients in the filtered image using a Prewitt filter
# prewitt_img = prewitt(gauss_img)
# prewitt_img = prewitt(med_img)
# prewitt_img = prewitt(gauss_med_img)
prewitt_img = prewitt(med_gauss_img)

# Use Otsu's thresholding method to compute a threshold, T, in the gradient image
thresh = threshold_otsu(prewitt_img)

# Apply the threshold, T, to the gradient image to create a binary image.
binary_img = prewitt_img > thresh

min_val = prewitt_img.min()
max_val = prewitt_img.max()

# Show the original image, the gradient image, and the binary image
fig, ax = plt.subplots(ncols=3)
ax[0].imshow(img, cmap="gray")
ax[1].imshow(prewitt_img, cmap="gray")
ax[2].imshow(binary_img, vmin=min_val, vmax=max_val, cmap="terrain")
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.show()