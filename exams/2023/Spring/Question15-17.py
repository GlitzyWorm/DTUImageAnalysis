### Setup ###
from skimage import io
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import disk, erosion, square
from skimage.filters import median

in_dir = "data/Letters/"

### Question 15-17 ###

# Load the image Letters.png
letters = io.imread(in_dir + "Letters.png")

# Extract the red, green and blue channels
red_channel = letters[:, :, 0]
green_channel = letters[:, :, 1]
blue_channel = letters[:, :, 2]

# Create a binary mask where all pixels with R > 100, G < 100 and B < 100 are set to 1, otherwise 0
binary_mask = ((red_channel > 100) & (green_channel < 100) & (blue_channel < 100)).astype(int)

# Erode the binary mask using a disk with a radius of 3 pixels
eroded_mask = erosion(binary_mask, disk(3))

# Count the number of pixels in the eroded mask
num_pixels = eroded_mask.sum()
print(num_pixels)

# Convert the input image to grayscale
gray_letters = rgb2gray(letters)

# Apply a median filter with a kernel size of 8x8 to the grayscale image
filtered_letters = median(gray_letters, square(8))

# Print the value at (100, 100) in the filtered image
print(f"The value at (100, 100) in the filtered image is: {filtered_letters[100, 100]}")

# Label the connected components in the eroded mask
labelled_mask = label(eroded_mask, connectivity=2)

# Compute the area and perimeter of all the BLOB's
regions = regionprops(labelled_mask)

# Remove all BLOB's with an area less than 1000 pixels or an area greater than 4000 pixels or a perimeter less than 300
filtered_regions = [region for region in regions if 1000 <= region.area <= 4000 and region.perimeter >= 300]

# Use the filtered regions to create a new binary mask
new_mask = sum([labelled_mask == region.label for region in filtered_regions])

# Plot the new mask
io.imshow(new_mask)
io.show()


