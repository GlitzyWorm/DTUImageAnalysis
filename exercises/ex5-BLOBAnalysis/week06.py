from skimage import io, color, morphology
from skimage.util import img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.filters import threshold_otsu
from skimage import segmentation
from skimage import measure
from skimage.color import label2rgb


def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis('off')
    io.show()


def show_comparison_2(original, cmap1, modified, modified_name, cmap2):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=cmap1)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified, cmap=cmap2)
    ax2.set_title(modified_name)
    ax2.axis('off')
    io.show()


# Exercise 1: Binary image from original image
# in_dir = "data/"
# im_name = "lego_4_small.png"
# img_org = io.imread(in_dir + im_name)
# img = color.rgb2gray(img_org)
# threshold = threshold_otsu(img)
# binary = img < threshold
# show_comparison(img_org, binary, 'Binary image')

# Exercise 2: Remove border BLOBs
# clean_border = segmentation.clear_border(binary)
# show_comparison_2(binary, 'gray', clean_border, 'Clean border', 'gray')

# Exercise 3: Cleaning using morphological operations
# closed = morphology.closing(clean_border, morphology.disk(5))
# opened = morphology.opening(closed, morphology.disk(5))
# show_comparison_2(clean_border, 'gray', opened, 'Opened', 'gray')

# Exercise 4: Labeling BLOBs
# label_img = measure.label(opened)
# n_labels = label_img.max()
# print(f"Number of labels: {n_labels}")

# Exercise 5: Visualize found labels
# image_label_overlay = label2rgb(label_img, image=img_org)
# show_comparison(img_org, image_label_overlay, 'Label overlay')

# Exercise 6: Compute BLOB features
# region_props = measure.regionprops(label_img)
# areas = np.array([prop.area for prop in region_props])
# plt.hist(areas, bins=50)
# plt.show()

# Exercise 7: Exploring BLOB features
# See Ex5-BlobAnalysisInteractive.py

# Pre-exercise 8
in_dir = "data/"
img_org = io.imread(in_dir + 'Sample E2 - U2OS DAPI channel.tiff')
# slice to extract smaller image
img_small = img_org[700:1200, 900:1400]
img_gray = img_as_ubyte(img_small)
# io.imshow(img_gray, vmin=0, vmax=50)
# plt.title('DAPI Stained U2OS cell nuclei')
# io.show()

# avoid bin with value 0 due to the very large number of background pixels
# plt.hist(img_gray.ravel(), bins=256, range=(1, 100))
# io.show()

# Exercise 8: Threshold selection
threshold = threshold_otsu(img_gray)
binary = img_gray > threshold
# show_comparison_2(img_small, None, binary, 'Binary image', 'binary')

# Exercise 9: Remove border BLOBS
clean_border = segmentation.clear_border(binary)

label_img = measure.label(clean_border)
image_label_overlay = label2rgb(label_img)
# show_comparison_2(img_org, None, image_label_overlay, 'Found BLOBS', None)

# Exercise 10: BLOB features
region_props = measure.regionprops(label_img)

# All areas
areas = np.array([prop.area for prop in region_props])

# Plot histogram of areas
# plt.hist(areas, bins=150)
# plt.show()

# Exercise 11: BLOB classification by area
min_area = 50
max_area = 150

# Create a copy of the label_img
label_img_filter = label_img
for region in region_props:
    # Find the areas that do not fit our criteria
    if region.area > max_area or region.area < min_area:
        # set the pixels in the invalid areas to background
        for cords in region.coords:
            label_img_filter[cords[0], cords[1]] = 0
# Create binary image from the filtered label image
i_area = label_img_filter > 0
# show_comparison(img_small, i_area, 'Found nuclei based on area')

# Exercise 12: Feature space
perimeters = np.array([prop.perimeter for prop in region_props])

# Plot area vs perimeter
# plt.scatter(areas, perimeters)
# plt.xlabel('Area')
# plt.ylabel('Perimeter')
# plt.show()

# Exercise 13: BLOB Circularity
# Compute circularity
circularities = np.array([4 * math.pi * prop.area / prop.perimeter ** 2 for prop in region_props])

# Plot histogram of circularities
# plt.hist(circularities, bins=50)
# plt.show()

# Only keep the BLOBS with circularity > 0.8
label_img_filter = label_img
for region in region_props:
    if 4 * math.pi * region.area / region.perimeter ** 2 < 0.8:
        for cords in region.coords:
            label_img_filter[cords[0], cords[1]] = 0
i_circularity = label_img_filter > 0
# show_comparison(img_small, i_circularity, 'Found nuclei based on circularity')

# Exercise 14: BLOB circularity and area
# Plot area vs circularity
# plt.scatter(areas, circularities)
# plt.xlabel('Area')
# plt.ylabel('Circularity')
# plt.show()

# Count the number of BLOBS in the circularity filtered image
label_img_filter = measure.label(i_circularity)
n_labels = label_img_filter.max()
# print(f"Number of labels: {n_labels}")

# Exercise 15: Large scale testing
from skimage import io, img_as_ubyte, measure, segmentation, filters
from skimage.color import label2rgb
import numpy as np
import math


def count_objects_by_area_and_circularity(image_path, min_area=50, max_area=150, min_circularity=0.8):
    # Load the image
    img_org = io.imread(image_path)
    img_gray = img_as_ubyte(img_org)

    # Apply Otsu's threshold to create a binary image
    threshold = filters.threshold_otsu(img_gray)
    binary = img_gray > threshold

    # Remove objects touching the image border
    clean_border = segmentation.clear_border(binary)

    # Label the objects in the cleaned binary image
    label_img = measure.label(clean_border)

    # Get properties of labeled regions
    region_props = measure.regionprops(label_img)

    # Initialize a new label image for the filtered objects
    label_img_filtered = np.zeros_like(label_img)

    # Object counter
    valid_obj_count = 0

    # Filter objects based on area and circularity
    for region in region_props:
        area = region.area
        perimeter = region.perimeter
        circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        if min_area <= area <= max_area and circularity >= min_circularity:
            valid_obj_count += 1
            for coords in region.coords:
                label_img_filtered[coords[0], coords[1]] = valid_obj_count

    return valid_obj_count, label_img_filtered


# Example usage
in_dir = "data/"
image_path = in_dir + 'Sample E2 - U2OS DAPI channel.tiff'
# count, label_img_filtered = count_objects_by_area_and_circularity(image_path)
# print(f"Number of objects meeting area and circularity criteria: {count}")
# image_label_overlay = label2rgb(label_img_filtered)
# io.imshow(image_label_overlay)
# io.show()

# Exercise 16: COS7 cell classification
# in_dir = "data/"
# image_path = in_dir + 'Sample G1 - COS7 cells DAPI channel.tiff'
# count, label_img_filtered = count_objects_by_area_and_circularity(image_path)
# print(f"Number of objects meeting area and circularity criteria: {count}")
# image_label_overlay = label2rgb(label_img_filtered)
# io.imshow(image_label_overlay)
# io.show()
