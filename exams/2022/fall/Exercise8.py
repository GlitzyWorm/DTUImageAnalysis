### Setup ###
import numpy as np
from skimage import io
from skimage.filters import prewitt

in_dir = "data/Filtering/"

### Exercise 8 ###

# Load the image rocket.png
img = io.imread(in_dir + "rocket.png", as_gray=True)

# Apply the prewitt filter to the image
prewitt_img = prewitt(img)

# Threshold the filereted image so all values above 0.06 are set to 1 and below to 0
thresholded = prewitt_img > 0.06

# Display the thresholded image (debugging)
# io.imshow(thresholded)
# io.show()

# Calculate the number of pixels with value 1
n_pixels = np.sum(thresholded == 1)

# Print the number of pixels
print(n_pixels)