### Setup ###
from skimage import io
from skimage.color import rgb2hsv
from skimage.filters.thresholding import threshold_otsu
from skimage.morphology import erosion, disk

in_dir = "data/PixelWiseOps/"

### Exercise 22 ###

# Load the image pixelwise.png
img = io.imread(in_dir + "pixelwise.png")

# Convert the image to HSV
img_hsv = rgb2hsv(img)

# Extract the saturation channel
saturation = img_hsv[:, :, 1]

# Threshold using Otsu's method
thresh = threshold_otsu(saturation)

# Create a binary mask using the threshold
binary_mask = saturation > thresh

# Perform a morphological erosion with a disk of radius 4 on the binary mask
eroded_mask = erosion(binary_mask, disk(4))

# Count the number of pixels in the eroded mask
n_pixels = eroded_mask.sum()

print(f"The number of pixels in the eroded mask is: {n_pixels}")