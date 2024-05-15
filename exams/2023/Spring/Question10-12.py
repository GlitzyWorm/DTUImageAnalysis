### Setup ###
import numpy as np
from skimage import io, img_as_ubyte
from skimage.transform import SimilarityTransform, warp, matrix_transform

in_dir = "data/LMRegistration/"

### Question 10 ###

# Load the images shoe_1.png and shoe_2.png
shoe_1 = io.imread(in_dir + "shoe_1.png")
shoe_2 = io.imread(in_dir + "shoe_2.png")

# Initialize the source and destination points
src = np.array([[40, 320], [425, 120], [740, 330]])
dst = np.array([[80, 320], [380, 155], [670, 300]])

# Compute the Similarity transform
tform = SimilarityTransform()
tform.estimate(src, dst)

# Transform the source points
src_transform = matrix_transform(src, tform.params)

# Print the scale factor
print(f"The scale factor is: {tform.scale}")

# Define an alignment error function
def compute_alignment_error(source, destination):
    distances = np.sum((source - destination) ** 2, axis=1)
    error = np.sum(distances)
    return error

# Calculate the alignment error F before transformation
F = compute_alignment_error(src, dst)

# Calculate the alignment error F after transformation
F_transformed = compute_alignment_error(src_transform, dst)

# Print the alignment errors
print(f"The alignment error F before transformation is: {F}")
print(f"The alignment error F after transformation is: {F_transformed}")
print(f"The difference in alignment error is: {np.abs(F - F_transformed)}")

# Warp the shoe_2 image using the estimated transformation
warped_shoe_1 = warp(shoe_1, tform.inverse)

# Convert both the shoe_2 and warped_shoe_1 images to ubyte
ubyte_shoe_2 = img_as_ubyte(shoe_2)
ubyte_warped_shoe_1 = img_as_ubyte(warped_shoe_1)

# Get the value at (200, 200) in the blue channel of both ubyte_warped_shoe_2 and ubyte_shoe_1
value_warped = ubyte_warped_shoe_1[200, 200, 2]
value_shoe = ubyte_shoe_2[200, 200, 2]

# Print the values
print(f"The value at (200, 200) in the warped image is: {value_warped}")
print(f"The value at (200, 200) in the original image is: {value_shoe}")

# Get the absolute difference between the two values
diff = np.abs(value_shoe - value_warped)

# Print the absolute difference
print(f"The absolute difference between the values is: {diff}")



