### Setup ###
import numpy as np
import skimage.filters.rank
from skimage import io, img_as_ubyte

in_dir = "data/"

# Load photo flowerwall.png from data folder
img = io.imread(in_dir + "flowerwall.png", as_gray=True)

### Q.8 ###

footprint = np.ones((15, 15))
img_average = skimage.filters.rank.mean(img_as_ubyte(img), footprint)

# What is the resulting pixel value in the pixel at row=5, column=50
# (when using a 1-based matrixbased coordinate system)?
resulting_pixel = img_average[5, 50]
print(f"Resulting pixel value: {resulting_pixel}")