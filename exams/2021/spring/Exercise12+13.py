### Setup ###
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.transform import SimilarityTransform, warp

in_dir = "data/"


### Q. 12 ###

# Load the seven landmarks (x, y) from the files catmovingPoints.text and catfixedPoints.txt
# and store them in the variables moving_points and fixed_points, respectively.

def read_landmark_file(file_name):
    f = open(file_name, 'r')
    lm_s = f.readline().strip().split(' ')
    n_lms = float(lm_s[0])
    if n_lms < 3:
        print(f"Not enough landmarks found")
        return None

    new_lms = 7
    # 3 landmarks each with (x,y)
    lm = np.zeros((new_lms, 2))
    for i in range(new_lms):
        lm[i, 0] = lm_s[0 + i * 2]
        lm[i, 1] = lm_s[1 + i * 2]
    return lm


# Load the moving points
moving_points = read_landmark_file(in_dir + "catmovingPoints.txt")

# Load the fixed points
fixed_points = read_landmark_file(in_dir + "catfixedPoints.txt")

# Calculate the sum of squared differences between the fixed and moving points
# and store it in the variable ssd.
ssd = np.sum((fixed_points - moving_points) ** 2)

# Print the sum of squared differences
print(ssd)

### Q. 13 ###

# Load the images cat1.png and cat2.png and store them in the variables im1 and im2, respectively.
# Read the images
im1 = io.imread(in_dir + "cat1.png")
im2 = io.imread(in_dir + "cat2.png")

# Apply a similarity transformation to the moving points to align them with the fixed points.
tform = SimilarityTransform()
tform.estimate(moving_points, fixed_points)
warped = warp(im2, tform.inverse)

# Display the images im1 and warped side by side.
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
ax1.imshow(im1)
ax1.set_title('Original')
ax1.axis('off')
ax2.imshow(warped)
ax2.set_title('Warped')
ax2.axis('off')
io.show()



