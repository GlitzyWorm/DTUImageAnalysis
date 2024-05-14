### Setup ###
from skimage import io, color
from skimage.filters.thresholding import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import regionprops, label

in_dir = "data/BLOBs/"

### Exercise 12 ###

# Load the image figures.png
img = io.imread(in_dir + "figures.png")

# Convert to grayscale
img_gray = color.rgb2gray(img)

# Compute Otsu's threshold
threshold = threshold_otsu(img_gray)

# Threshold the image. All values below the threshold should be set to 1, otherwise 0
img_threshold = img_gray < threshold

# Remove all BLOB's connected to the border of the image
img_no_border = clear_border(img_threshold)

# Compute the area and perimeter of the BLOB's in the image
img_label = label(img_no_border)
regions = regionprops(img_label)

# Count the number of BLOB's with an area greater than 13000 pixels
n_blobs = 0
for region in regions:
    if region.area > 13000:
        n_blobs += 1

# Print the number of BLOB's
print(n_blobs)


### Exercise 13 ###

# Find the BLOB with the largest area and print its perimeter
max_area = 0
max_perimeter = 0
for region in regions:
    if region.area > max_area:
        max_area = region.area
        max_perimeter = region.perimeter

# Print the perimeter of the BLOB with the largest area
print(max_perimeter)

