### Setup ###
import numpy as np
from skimage import io, color, img_as_ubyte
from skimage._shared.filters import gaussian
from skimage.transform import EuclideanTransform, matrix_transform, warp, rotate

in_dir = "data/GeomTrans/"

### Exercise 17 ###

# Source landmarks
src = np.array([[220, 55], [105, 675], [315, 675]])

# Destination landmarks
dst = np.array([[100, 165], [200, 605], [379, 525]])


# Compute squared Euclidean distances and sum them
def compute_alignment_error(source, destination):
    distances = np.sum((source - destination) ** 2, axis=1)
    error = np.sum(distances)
    return error


### Answer to Exercise 18 ##
# Calculate the alignment error F
F = compute_alignment_error(src, dst)
print(f"The alignment error F is: {F}")

# Compute the Euclidean transform that brings the source landmarks to the destination landmarks
tform = EuclideanTransform()
tform.estimate(src, dst)
src_transform = matrix_transform(src, tform.params)

### Answer to Exercise 19 ###
# Check the alignment error after the transformation
F_transformed = compute_alignment_error(src_transform, dst)
print(f"The alignment error F after transformation is: {F_transformed}")

# Load rocket.png
rocket = io.imread(in_dir + "rocket.png")

# Convert the image to grayscale
rocket_gray = color.rgb2gray(rocket)

# Transform the image using the estimated transformation
transformed_rocket = warp(rocket_gray, tform.inverse)

# Convert the transformed image to ubyte
ubyte_rocket = img_as_ubyte(transformed_rocket)

### Answer to Exercise 17 ###
# Print the value at (150, 150) in the warped image
print(f"The value at (150, 150) in the warped image is: {ubyte_rocket[150, 150]}")

# Apply a Gaussian filter to the rocket_gray with a sigma of 3
rocket_gauss = gaussian(rocket_gray, 3)

# Convert back to ubyte
ubyte_rocket_gauss = img_as_ubyte(rocket_gauss)

### Answer to Exercise 20 ###
# Print the value at (100, 100) in the Gaussian filtered image
print(f"The value at (100, 100) in the Gaussian filtered image is: {ubyte_rocket_gauss[100, 100]}")

### Exercise 21 ###
# Load the image CPHSun.png
CPH_sun = io.imread(in_dir + "CPHSun.png")

# Rotate the image 16 degrees with a rotation center of (20, 20)
rotation_angle = 16
rotation_center = (20, 20)
rotated_CPH_sun = rotate(CPH_sun, rotation_angle, center=rotation_center)

# Convert the rotated image to ubyte
ubyte_rotated_CPH_sun = img_as_ubyte(rotated_CPH_sun)

# Print the value at (200, 200) in the rotated image
print(f"The value at (200, 200) in the rotated image is: {ubyte_rotated_CPH_sun[200, 200]}")
