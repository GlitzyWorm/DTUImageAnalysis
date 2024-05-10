### Setup ###
import numpy as np
from skimage import io
from skimage.color import rgb2hsv
from skimage.morphology import disk, opening

in_dir = "data/"

### Q.5 ###

# Load photo flower.png from data folder
img = io.imread(in_dir + "flower.png")

# Convert from RGB to HSV
img_hsv = rgb2hsv(img)

# Perform a threshold on the hue channel with the limits H < 0.25, S > 0.8 and V > 0.8
img_thresh = np.zeros((img.shape[0], img.shape[1]))

h_comp = img_hsv[:, :, 0]
s_comp = img_hsv[:, :, 1]
v_comp = img_hsv[:, :, 2]

img_thresh[(h_comp < 0.25) & (s_comp > 0.8) & (v_comp > 0.8)] = 1

# Morphological opening using a disk-shaped structuring element with a radius of 5
img_open = opening(img_thresh, disk(5))

# How many foreground pixels are there in the opened image?
n_foreground = np.sum(img_open)
print(f"Number of foreground pixels in the opened image: {n_foreground}")


