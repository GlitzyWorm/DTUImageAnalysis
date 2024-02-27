import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
from skimage import io, color
from skimage.filters import threshold_otsu


# From https://scikit-image.org/docs/stable/auto_examples/applications/plot_morphology.html
def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    io.show()


# Plot comparision with 3 images
def plot_comparison_3(original, filtered1, filtered2, filter_name1, filter_name2):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered1, cmap=plt.cm.gray)
    ax2.set_title(filter_name1)
    ax2.axis('off')
    ax3.imshow(filtered2, cmap=plt.cm.gray)
    ax3.set_title(filter_name2)
    ax3.axis('off')
    io.show()


def plot_comparison_4(original, filtered1, filtered2, filtered3, filter_name1, filter_name2, filter_name3):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered1, cmap=plt.cm.gray)
    ax2.set_title(filter_name1)
    ax2.axis('off')
    ax3.imshow(filtered2, cmap=plt.cm.gray)
    ax3.set_title(filter_name2)
    ax3.axis('off')
    ax4.imshow(filtered3, cmap=plt.cm.gray)
    ax4.set_title(filter_name3)
    ax4.axis('off')
    io.show()


def compute_outline(bin_img_def):
    """
    Computes the outline of a binary image
    """
    footprint_def = disk(1)
    dilated = dilation(bin_img_def, footprint_def)
    outline = np.logical_xor(dilated, bin_img_def)
    return outline


def compute_outline_with_disk(bin_img_def, disk_size):
    """
    Computes the outline of a binary image
    """
    footprint_def = disk(disk_size)
    dilated = dilation(bin_img_def, footprint_def)
    outline = np.logical_xor(dilated, bin_img_def)
    return outline

# Exercise 1: Image morphology on a single object

# Load lego_5.png
# img = io.imread("data/lego_5.png")

# Convert to grayscale
# img_gray = color.rgb2gray(img)

# Threshold the image
# thresh = threshold_otsu(img_gray)

# Apply the threshold to the image and generate a binary image bin_img
# bin_img = img_gray < thresh

# Visualize the original and binary images
# plot_comparison(img_gray, bin_img, 'Binary image')

# Pre-exercise 2 & 3:
footprint = disk(7)

# Check the size and shape of the structuring element
# print(footprint)

# Exercise 2: Erosion
# eroded = erosion(bin_img, footprint)
# plot_comparison(bin_img, eroded, 'erosion')

# Exercise 3: Dilation
# dilated = dilation(bin_img, footprint)
# plot_comparison(bin_img, dilated, 'dilation')

# Exercise 4: Opening
# opened = opening(bin_img, footprint)
# plot_comparison(bin_img, opened, 'opening')

# Exercise 5: Closing
# closed = closing(bin_img, footprint)
# plot_comparison(bin_img, closed, 'closing')

# Exercise 6: Compute the outline of the binary image of the lego brick. What do you observe?
# outline = compute_outline(bin_img)
# plot_comparison(bin_img, outline, 'outline')

# Exercise 7:
# Do an opening with a disk of size 1 on the binary lego image.
# opened = opening(bin_img, disk(1))

# Do a closing with a disk of size 15 on the result of the opening.
# closed = closing(opened, disk(15))

# Compute the outline of the binary image of the lego brick.
# outline = compute_outline(closed)
# plot_comparison(bin_img, outline, 'outline')

# Only the outline of the lego brick is visible in the resulting image.
# The opening removes the small white spots inside the lego brick,
# and the closing removes the small black spots outside the lego brick.

# Exercise 8: Morphology on multiple objects
# Load lego_7.png
# img = io.imread("data/lego_7.png")

# Convert to grayscale
# img_gray = color.rgb2gray(img)

# Threshold the image
# thresh = threshold_otsu(img_gray)

# Apply the threshold to the image and generate a binary image bin_img
# bin_img = img_gray < thresh

# Visualize the original and binary images
# plot_comparison(img_gray, bin_img, 'Binary image')

# Exercise 9: We would like to find a way so only the outline of the entire brick is computed

# Do a close operation with a disk of size 15 on the binary lego image.
# closed = closing(bin_img, disk(8))

# Compute the outline of the binary image of the lego brick.
# outline = compute_outline_with_disk(closed, 10)
# plot_comparison_3(bin_img, closed, outline, 'closed', 'outline')

# Exercise 10: Do the same for lego_3.png
# Load lego_3.png
# img = io.imread("data/lego_3.png")

# Convert to grayscale
# img_gray = color.rgb2gray(img)

# Threshold the image
# thresh = threshold_otsu(img_gray)

# Apply the threshold to the image and generate a binary image bin_img
# bin_img = img_gray < thresh

# Visualize the original and binary images
# plot_comparison(img_gray, bin_img, 'Binary image')

# Do a close operation with a disk of size 15 on the binary lego image.
# closed = closing(bin_img, disk(20))

# Compute the outline of the binary image of the lego brick.
# outline = compute_outline_with_disk(closed, 10)

# Visualize the original, closed, and outline images
# plot_comparison_3(bin_img, closed, outline, 'closed', 'outline')

# Exercise 11 & 12: Morphology on multiple connected objects

# Load lego_9.png
# img = io.imread("data/lego_9.png")

# Convert to grayscale
# img_gray = color.rgb2gray(img)

# Threshold the image
# thresh = threshold_otsu(img_gray)

# Apply the threshold to the image and generate a binary image bin_img
# bin_img = img_gray < thresh

# Do a close operation with a disk of size 15 on the binary lego image.
# closed = closing(bin_img, disk(5))

# Compute the outline of the binary image of the lego brick.
# outline = compute_outline_with_disk(closed, 5)

# Visualize the original, closed, and outline images
# plot_comparison_3(bin_img, closed, outline, 'closed', 'outline')

# Exercise 13: Seperate the connected objects with erosion
# eroded = erosion(closed, disk(50))

# Compute the outline of the binary image of the lego brick.
# outline = compute_outline_with_disk(eroded, 5)

# Visualize the original, eroded, and outline images
# plot_comparison_4(bin_img, closed, eroded, outline, 'closed', 'eroded', 'outline')

# Exercise 14: Make the objects larger again using dilate
# dilated = dilation(eroded, disk(15))

# Compute the outline of the binary image of the lego brick.
# outline = compute_outline_with_disk(dilated, 5)

# Visualize the original, dilated, and outline images
# plot_comparison_4(bin_img, eroded, dilated, outline, 'eroded', 'dilated', 'outline')

# Exercise 15: Puzzle piece analysis
# Load puzzle_pieces.png
img = io.imread("data/puzzle_pieces.png")

# Convert to grayscale
img_gray = color.rgb2gray(img)

# Threshold the image
thresh = threshold_otsu(img_gray)

# Apply the threshold to the image and generate a binary image bin_img
bin_img = img_gray < thresh

# Visualize the original and binary images
# plot_comparison(img_gray, bin_img, 'Binary image')

# Exercise 16: Compute the outline of the binary image of the puzzle pieces

# Do a closing operation on the binary puzzle pieces image.
closed = closing(bin_img, disk(20))

# Compute the outline of the binary image of the puzzle pieces.
outline = compute_outline_with_disk(closed, 5)

# Visualize the original, opened, and outline images
plot_comparison_3(bin_img, closed, outline, 'closed', 'outline')