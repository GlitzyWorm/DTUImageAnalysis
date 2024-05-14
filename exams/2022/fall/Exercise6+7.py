### Setup ###
from skimage import io, color
from skimage.filters.thresholding import threshold_otsu

in_dir = "data/PixelWiseOps/"

### Exercise 6 ###

# Load the image pixelwise.png
img = io.imread(in_dir + "pixelwise.png")

# Convert to grayscale
img_grey = color.rgb2gray(img)

# Linear grayscale transformation with minimum pixel value of 0.1 and maximum pixel value of 0.6
img_grey_transformed = (img_grey - img_grey.min()) / (img_grey.max() - img_grey.min()) * 0.5 + 0.1

# Compute a pixel value threshold using Otsu's method
threshold = threshold_otsu(img_grey_transformed)

# Apply the threshold to the transformed image so all values below the threshold are set to 1 and above to 0
thresholded = img_grey_transformed > threshold

# Display the thresholded image
io.imshow(thresholded)
io.show()

### Exercise 7 ###

# Print the threshold value
print(threshold)
