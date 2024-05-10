### Setup ###
import numpy as np
from skimage import io, color

in_dir = "data/"

### Q.3 ###

# Load photo sky_gray.png from data folder
img = io.imread(in_dir + "sky_gray.png")

# Plot the image
# plt.imshow(img, cmap="gray")
# plt.show()

# Peform a linear histogram stretch so the maximum pixel value is 200 and the minimum is 10
img_stretch = 190/(np.max(img) - np.min(img)) * (img - np.min(img)) + 10

# Print the minimum and maximum pixel values of the stretched image
print(f"Minimum pixel value of the stretched image: {np.min(img_stretch)}")
print(f"Maximum pixel value of the stretched image: {np.max(img_stretch)}")

# Print the average pixel value of the stretched image
print(f"Average pixel value of the stretched image: {np.mean(img_stretch)}")