### Setup ###
import numpy as np
from skimage import io, measure

in_dir = 'data/'

### Q. 10 ###

# Load books_bw.png
img = io.imread(in_dir + 'books_bw.png', as_gray=True)

# Find all BLOB's using 8-connectivity
label_img = measure.label(img, connectivity=2)

# Compute the area and perimeter of all BLOB's
props = measure.regionprops_table(label_img, properties=('label', 'area', 'perimeter'))

# Convert to a numpy array for easy filtering
areas = np.array(props['area'])
perimeters = np.array(props['perimeter'])
labels = np.array(props['label'])

# Filter blobs based on area and perimeter conditions
filter_mask = (areas > 100) & (perimeters > 500)
filtered_labels = labels[filter_mask]

# Create an output image showing only the filtered blobs
filtered_image = np.isin(label_img, filtered_labels)

# Display the filtered image
io.imshow(filtered_image)
io.show()
