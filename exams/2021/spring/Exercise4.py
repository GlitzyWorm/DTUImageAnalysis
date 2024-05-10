### Setup ###
import numpy as np
from skimage import io
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk

in_dir = "data/"

### Q.4 ###

# Load photo sky.png from data folder
img = io.imread(in_dir + "sky.png")

# Do an RGB threshold with the limits R < 100, G > 85, G < 200 and B > 150,
# where values within these limits are set to foreground and the rest to background
img_thresh = np.zeros((img.shape[0], img.shape[1]))

r_comp = img[:, :, 0]
g_comp = img[:, :, 1]
b_comp = img[:, :, 2]

img_thresh[(r_comp < 100) & (g_comp > 85) & (g_comp < 200) & (b_comp > 150)] = 1

# Morphological erode using a disk.shaped structuring element with a radius of 5
img_erode = erosion(img_thresh, disk(5))

# How many foreground pixels are there in the eroded image?
n_foreground = np.sum(img_erode)
print(f"Number of foreground pixels in the eroded image: {n_foreground}")