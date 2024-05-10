### Setup ###
import numpy as np
from skimage import io, img_as_float, img_as_ubyte
from skimage.morphology import disk, square
from skimage.filters import median

in_dir = "data/"

### Helper function ###
def gamma_map(img, gamma):
    """
    Gamma mapping of an image
    :param img: Input image
    :param gamma: Gamma value
    :return: Image, where the gamma mapping is applied
    """
    # img_as_float will divide all pixel values with 255.0
    img_float = img_as_float(img)

    # Do something here
    img_out = np.power(img_float, gamma)
    # img_out = img_float ** gamma

    # img_as_ubyte will multiply all pixel values with 255.0 before converting to unsigned byte
    return img_as_ubyte(img_out)

### Q.7 ###

# Load photo sky_gray.png from data folder
img = io.imread(in_dir + "sky_gray.png", as_gray=True)

# Transform the image using a gamma mapping with gamma = 1.21
gamma = 1.21
img_gamma = gamma_map(img, gamma)

# Filter the image using a 5x5 median filter
footprint = np.ones((5, 5))
img_median = median(img_gamma, footprint)

# What is the resulting pixel value in the pixel at row=40, column=50
# (when using a 1-based matrixbased coordinate system)?
resulting_pixel = img_median[40, 50]
print(f"Resulting pixel value: {resulting_pixel}")