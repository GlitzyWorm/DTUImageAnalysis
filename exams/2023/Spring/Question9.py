### Setup ###
from skimage import io
from skimage.color import rgb2hsv
from skimage.morphology import dilation, disk

in_dir = "data/Pixelwise/"

### Question 9 ###

# Load the image nike.png
nike = io.imread(in_dir + "nike.png")

# Convert to HSV
nike_hsv = rgb2hsv(nike)

# Extract the hue channel
hue = nike_hsv[:, :, 0]

# Create a binary mask for the hue channel that is 1 where the hue is between 0.3 and 0.7, and 0 elsewhere
binary_mask = ((hue >= 0.3) & (hue <= 0.7)).astype(int)

# Perform a morphological dilation with a disk of radius 8
dilated_mask = dilation(binary_mask, disk(8))

# Count the number of pixels in the dilated mask
num_pixels = dilated_mask.sum()

# Print the number of pixels
print(num_pixels)

