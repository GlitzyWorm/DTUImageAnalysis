### Setup ###
from skimage import io
from skimage.color import rgb2gray
from skimage.filters.thresholding import threshold_otsu
from skimage.transform import rotate

in_dir = "data/GeomTrans/"

### Question 24-25 ###

# Load the image lights.png
lights = io.imread(in_dir + "lights.png")

# Rotate the image 11 degrees with a rotation center at (40, 40)
rotation_angle = 11
rotation_center = (40, 40)
rotated_lights = rotate(lights, rotation_angle, center=rotation_center)

# Convert the rotated image to grayscale
rotated_lights_gray = rgb2gray(rotated_lights)

# Compute the threshold using Otsu's method
threshold = threshold_otsu(rotated_lights_gray)

# Create a binary mask using the threshold
binary_mask = rotated_lights_gray > threshold

# Compute the percentage of pixels that are set to 1 in the binary mask
percentage = binary_mask.sum() / binary_mask.size * 100
print(f"The percentage of pixels that are set to 1 in the binary mask is: {percentage:.2f}%")

# Print the threshold value
print(f"The threshold value is: {threshold}")
