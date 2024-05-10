### Setup ###
import numpy as np
from skimage import io, segmentation, color, img_as_ubyte, measure
from skimage.color import rgb2gray
from skimage.filters.thresholding import threshold_otsu
from skimage.morphology import disk, closing, opening

in_dir = "data/"

### Q.9 ###
# Load photo floorboards.png from data folder
img = io.imread(in_dir + "floorboards.png")
img = img_as_ubyte(img)

# Apply a threshold of 100, so pixels below the threshold are set to foreground
# and the rest is set to background.
threshold = 100  # threshold_otsu(img)
print(f"Threshold value: {threshold}")
img_thresh = img < threshold
img_thresh = img_as_ubyte(color.rgb2gray(img_thresh))

# Plot the thresholded image
io.imshow(img_thresh)
io.show()

# To remove noise a morphological closing is performed with a disk-shaped structuring
# element with radius=10
img_close = closing(img_thresh, disk(10))

# Then followed by a morphological opening with a disk-shaped structuring element with radius=3
img_open = opening(img_close, disk(3))

# Finally, all BLOB's that are connected to the image border are removed
clean_border = segmentation.clear_border(img_open)

# How many foreground pixels are there in the cleaned image?
n_foreground = np.sum(clean_border)
print(f"Number of foreground pixels in the cleaned image: {n_foreground}")

# Get the number of pixels that is different from 0
n_pixels = np.sum(clean_border != 0)
print(f"Number of pixels in the cleaned image: {n_pixels}")

### Q.10 ###

# Find all BLOB's using 8-connectivity
label_img = measure.label(clean_border, connectivity=2)

# Compute the area of all BLOB's
props = measure.regionprops(label_img)
areas = [prop.area for prop in props]
print(f"Areas of BLOB's: {areas}")

# Only keep BLOB's with an area larger than 100 pixels
large_areas = [area for area in areas if area > 100]
print(f"Areas of BLOB's larger than 100 pixels: {large_areas}")

# Compute the number of BLOB's
n_blobs = len(large_areas)
print(f"Number of BLOB's larger than 100 pixels: {n_blobs}")
