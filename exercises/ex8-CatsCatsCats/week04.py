import random

from skimage import io
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.decomposition import PCA
from skimage.transform import SimilarityTransform
from skimage.transform import warp
import os
import pathlib


def read_landmark_file(file_name):
    f = open(file_name, 'r')
    lm_s = f.readline().strip().split(' ')
    n_lms = int(lm_s[0])
    if n_lms < 3:
        print(f"Not enough landmarks found")
        return None

    new_lms = 3
    # 3 landmarks each with (x,y)
    lm = np.zeros((new_lms, 2))
    for i in range(new_lms):
        lm[i, 0] = lm_s[1 + i * 2]
        lm[i, 1] = lm_s[2 + i * 2]
    return lm


def align_and_crop_one_cat_to_destination_cat(img_src, lm_src, img_dst, lm_dst):
    """
    Landmark based alignment of one cat image to a destination
    :param img_src: Image of source cat
    :param lm_src: Landmarks for source cat
    :param lm_dst: Landmarks for destination cat
    :return: Warped and cropped source image. None if something did not work
    """
    tform = SimilarityTransform()
    tform.estimate(lm_src, lm_dst)
    warped = warp(img_src, tform.inverse, output_shape=img_dst.shape)

    # Center of crop region
    cy = 185
    cx = 210
    # half the size of the crop box
    sz = 180
    warp_crop = warped[cy - sz:cy + sz, cx - sz:cx + sz]
    shape = warp_crop.shape
    if shape[0] == sz * 2 and shape[1] == sz * 2:
        return img_as_ubyte(warp_crop)
    else:
        print(f"Could not crop image. It has shape {shape}. Probably to close to border of image")
        return None


def preprocess_all_cats(in_dir, out_dir):
    """
    Create aligned and cropped version of image
    :param in_dir: Where are the original photos and landmark files
    :param out_dir: Where should the preprocessed files be placed
    """
    dst = "data/ModelCat"
    dst_lm = read_landmark_file(f"{dst}.jpg.cat")
    dst_img = io.imread(f"{dst}.jpg")

    all_images = glob.glob(in_dir + "*.jpg")
    for img_idx in all_images:
        name_no_ext = os.path.splitext(img_idx)[0]
        base_name = os.path.basename(name_no_ext)
        out_name = f"{out_dir}/{base_name}_preprocessed.jpg"

        src_lm = read_landmark_file(f"{name_no_ext}.jpg.cat")
        src_img = io.imread(f"{name_no_ext}.jpg")

        proc_img = align_and_crop_one_cat_to_destination_cat(src_img, src_lm, dst_img, dst_lm)
        if proc_img is not None:
            io.imsave(out_name, proc_img)


def preprocess_one_cat():
    src = "data/MissingCat"
    dst = "data/ModelCat"
    out = "data/MissingCatProcessed.jpg"

    src_lm = read_landmark_file(f"{src}.jpg.cat")
    dst_lm = read_landmark_file(f"{dst}.jpg.cat")

    src_img = io.imread(f"{src}.jpg")
    dst_img = io.imread(f"{dst}.jpg")

    src_proc = align_and_crop_one_cat_to_destination_cat(src_img, src_lm, dst_img, dst_lm)
    if src_proc is None:
        return

    io.imsave(out, src_proc)

    fig, ax = plt.subplots(ncols=3, figsize=(16, 6))
    ax[0].imshow(src_img)
    ax[0].plot(src_lm[:, 0], src_lm[:, 1], '.r', markersize=12)
    ax[1].imshow(dst_img)
    ax[1].plot(dst_lm[:, 0], dst_lm[:, 1], '.r', markersize=12)
    ax[2].imshow(src_proc)
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()


def create_u_byte_image_from_vector(im_vec, height, width, channels):
    min_val = im_vec.min()
    max_val = im_vec.max()

    # Transform to [0, 1]
    im_vec = np.subtract(im_vec, min_val)
    im_vec = np.divide(im_vec, max_val - min_val)
    im_vec = im_vec.reshape(height, width, channels)
    im_out = img_as_ubyte(im_vec)
    return im_out


# Exercise 1: Preprocess all image in the training set.
# The preprocessing steps are:
#
# 1. Define a model cat (ModelCat.jpg) with associated landmarks (ModelCat.jpg.cat)
# 2. For each cat in the training data:
#   a. Use landmark based registration with a similarity transform to register the photo to the model cat
#   b. Crop the registered photo
#   c. Save the result in a fold called preprocessed
in_dir = "data/training_data/"
out_dir = "data/preprocessed/"
# preprocess_all_cats(in_dir, out_dir)

# Exercise 2: Compute the data matrix.

# Get the number of images in the preprocessed folder
all_images = glob.glob(out_dir + "*.jpg")
n_samples = len(all_images)

# Read the first image to get the size
img = io.imread(all_images[0])
height, width, channels = img.shape

# Set n_features to height * width * channels
n_features = height * width * channels

# Create empty data matrix
data_matrix = np.zeros((n_samples, n_features))

# Read the image files one by one and use flatten() to make each image into a 1-D vector (flat_img)
for i in range(n_samples):
    img = io.imread(all_images[i])
    flat_img = img.flatten()
    data_matrix[i, :] = flat_img

# Exercise 3: Compute the average cat.

# Compute the average cat by taking the mean of the data matrix
avg_cat = np.mean(data_matrix, axis=0)

# Exercise 4: Visualize the Mean Cat
# Create a mean cat image from the average cat vector
# mean_cat_img = create_u_byte_image_from_vector(avg_cat, height, width, channels)
# io.imshow(mean_cat_img)
# io.show()

# Exercise 6: Use the preprocess_one_cat function to preprocess the photo of the poor missing cat
# preprocess_one_cat()

# Exercise 7: Flatten the pixel values of the missing cat so it becomes a vector of values.
# Read the missing cat image
missing_cat = io.imread("data/MissingCatProcessed.jpg")
# Flatten the image
missing_cat_flat = missing_cat.flatten()

# Exercise 8: Subtract you missing cat data from all the rows in the data_matrix and
# for each row compute the sum of squared differences. This can for example be done by:

# Subtract the missing cat from the data matrix
# sub_data = data_matrix - missing_cat_flat
# Compute the sum of squared differences
# sub_distances = np.linalg.norm(sub_data, axis=1)

# Exercise 9: Find the cat that looks most like your missing cat by finding the cat, where the SSD is smallest.
# You can for example use np.argmin.

# Find the index of the cat that looks most like the missing cat
# idx = np.argmin(sub_distances)

# Exercise 10: Extract the found cat from the data_matrix and use create_u_byte_image_from_vector to create an image
# that can be visualized. Did you find a good replacement cat? Do you think your neighbour will notice?
# Even with their glasses on?

# Extract the found cat from the data_matrix
# found_cat = data_matrix[idx, :]

# Create an image from the found cat
# found_cat_img = create_u_byte_image_from_vector(found_cat, height, width, channels)

# Visualize the found cat
# io.imshow(found_cat_img)
# io.show()

# Exercise 11: You can use np.argmax to find the cat that looks the least like the missing cat.

# Find the index of the cat that looks the least like the missing cat
# idx = np.argmax(sub_distances)

# Visualize the least similar cat
# least_similar_cat = data_matrix[idx, :]
# least_similar_cat_img = create_u_byte_image_from_vector(least_similar_cat, height, width, channels)
# io.imshow(least_similar_cat_img)
# io.show()

# Exercise 12: Start by computing the first 50 principal components:
print("Computing PCA")
cats_pca = PCA(n_components=50)
cats_pca.fit(data_matrix)

# Exercise 13: Plot the amount of the total variation explained by each component as function of the component number.
# Use the explained_variance_ratio_ attribute of the PCA object to do this.
# plt.plot(cats_pca.explained_variance_ratio_)
# plt.xlabel('Number of components')
# plt.ylabel('Explained variance')
# plt.show()

# Exercise 14: How much of the total variation is explained by the first component?
# print(f"The first component explains {cats_pca.explained_variance_ratio_[0] * 100:.2f}% of the total variation")

# Exercise 15: Project the cat images into PCA space:
components = cats_pca.transform(data_matrix)

# Exercise 16: Plot the PCA space by plotting all the cats first and second PCA coordinates in a (x, y) plot
# plt.scatter(components[:, 0], components[:, 1])
# plt.xlabel('1st component')
# plt.ylabel('2nd component')
# plt.show()

# Exercise 17: Use np.argmin and np.argmax to find the ids of the cats that have extreme values in the
# first and second PCA coordinates. Extract the cats data from the data matrix and use create_u_byte_image_from_vector
# to visualize these cats.
# Also plot the PCA space where you plot the extreme cats with another marker and color.

# Find the index of the cat that has the smallest value in the first PCA coordinate
# idx_small_first = np.argmin(components[:, 0])
# Extract the cat from the data matrix
# cat = data_matrix[idx_small_first, :]
# Create an image from the cat
# cat_img = create_u_byte_image_from_vector(cat, height, width, channels)
# Visualize the cat
# io.imshow(cat_img)
# io.show()

# Find the index of the cat that has the largest value in the first PCA coordinate
# idx_large_first = np.argmax(components[:, 0])
# Extract the cat from the data matrix
# cat = data_matrix[idx_large_first, :]
# Create an image from the cat
# cat_img = create_u_byte_image_from_vector(cat, height, width, channels)
# Visualize the cat
# io.imshow(cat_img)
# io.show()

# Find the index of the cat that has the smallest value in the second PCA coordinate
# idx_small_second = np.argmin(components[:, 1])
# Extract the cat from the data matrix
# cat = data_matrix[idx_small_second, :]
# Create an image from the cat
# cat_img = create_u_byte_image_from_vector(cat, height, width, channels)
# Visualize the cat
# io.imshow(cat_img)
# io.show()

# Find the index of the cat that has the largest value in the second PCA coordinate
# idx_large_second = np.argmax(components[:, 1])
# Extract the cat from the data matrix
# cat = data_matrix[idx_large_second, :]
# Create an image from the cat
# cat_img = create_u_byte_image_from_vector(cat, height, width, channels)
# Visualize the cat
# io.imshow(cat_img)
# io.show()

# Plot the PCA space where you plot the extreme cats with another marker and color
# plt.scatter(components[:, 0], components[:, 1])
# plt.scatter(components[idx_small_first, 0], components[idx_small_first, 1], color='r', marker='x')
# plt.scatter(components[idx_large_first, 0], components[idx_large_first, 1], color='g', marker='x')
# plt.scatter(components[idx_small_second, 0], components[idx_small_second, 1], color='b', marker='x')
# plt.scatter(components[idx_large_second, 0], components[idx_large_second, 1], color='y', marker='x')
# plt.xlabel('1st component')
# plt.ylabel('2nd component')
# plt.show()

# Exercise 18: How do these extreme cat photo look like? Are some actually of such bad quality that they should be
# removed from the training set?
# If you remove images from the training set,
# then you should run the PCA again. Do this until you are satisfied with the quality of the training data.

# Exercise 19: Create your first fake cat using the average image and the first principal component.
# You should choose experiment with different weight values (w) :
# w = 200000
# fake_cat = avg_cat + w * cats_pca.components_[0, :]
# fake_cat_img = create_u_byte_image_from_vector(fake_cat, height, width, channels)
# io.imshow(fake_cat_img)
# io.show()

# Exercise 21: Synthesize some cats, where you use both the first and
# second principal components and select their individual weights based on the PCA plot.

# w1 = -20
# w2 = 200
# fake_cat = avg_cat + w1 * cats_pca.components_[0, :] + w2 * cats_pca.components_[1, :]
# fake_cat_img = create_u_byte_image_from_vector(fake_cat, height, width, channels)
# io.imshow(fake_cat_img)
# io.show()

# Exercise 22: Synthesize and visualize cats that demonstrate the first three major modes of variation.
# Try show the average cat in the middle of a plot, with the negative sample to the left and the positive to the right.
# Can you recognise some visual patterns in these modes of variation?

# def synth_cat_plus(avg, pca, mode):
#     return avg + 3 * np.sqrt(pca.explained_variance_[mode]) * pca.components_[mode, :]
#
#
# def synth_cat_minus(avg, pca, mode):
#     return avg - 3 * np.sqrt(pca.explained_variance_[mode]) * pca.components_[mode, :]

# Average cat
# avg_cat_img = create_u_byte_image_from_vector(avg_cat, height, width, channels)

# Synthesize and visualize cats that demonstrate the first three major modes of variation. Create an image from the cat.
# fake_cat_img1 = create_u_byte_image_from_vector(synth_cat_plus(avg_cat, cats_pca, 1), height, width, channels)
# fake_cat_img2 = create_u_byte_image_from_vector(synth_cat_minus(avg_cat, cats_pca, 1), height, width, channels)

# Show fake_cat_img1, fake_cat_img2, fake_cat_img3 in a plot together
# fig, ax = plt.subplots(ncols=3, figsize=(16, 6))
# ax[0].imshow(fake_cat_img1)
# ax[1].imshow(avg_cat_img)
# ax[2].imshow(fake_cat_img2)
# for a in ax:
#     a.axis('off')
# plt.tight_layout()
# plt.show()

# Synthesize and visualize cats that demonstrate the first three major modes of variation. Create an image from the cat.
# fake_cat_img1 = create_u_byte_image_from_vector(synth_cat_plus(avg_cat, cats_pca, 2), height, width, channels)
# fake_cat_img2 = create_u_byte_image_from_vector(synth_cat_minus(avg_cat, cats_pca, 2), height, width, channels)

# Show fake_cat_img1, fake_cat_img2, fake_cat_img3 in a plot together
# fig, ax = plt.subplots(ncols=3, figsize=(16, 6))
# ax[0].imshow(fake_cat_img1)
# ax[1].imshow(avg_cat_img)
# ax[2].imshow(fake_cat_img2)
# for a in ax:
#     a.axis('off')
# plt.tight_layout()
# plt.show()

# Synthesize and visualize cats that demonstrate the first three major modes of variation. Create an image from the cat.
# fake_cat_img1 = create_u_byte_image_from_vector(synth_cat_plus(avg_cat, cats_pca, 3), height, width, channels)
# fake_cat_img2 = create_u_byte_image_from_vector(synth_cat_minus(avg_cat, cats_pca, 3), height, width, channels)

# Show fake_cat_img1, fake_cat_img2, fake_cat_img3 in a plot together
# fig, ax = plt.subplots(ncols=3, figsize=(16, 6))
# ax[0].imshow(fake_cat_img1)
# ax[1].imshow(avg_cat_img)
# ax[2].imshow(fake_cat_img2)
# for a in ax:
#     a.axis('off')
# plt.tight_layout()
# plt.show()

# Exercise 23: Generate as many cat photos as your heart desires.
# n_components_to_use = 50
# synth_cat = avg_cat
# for idx in range(n_components_to_use):
#     w = random.uniform(-1, 1) * 3 * np.sqrt(cats_pca.explained_variance_[idx])
#     synth_cat = synth_cat + w * cats_pca.components_[idx, :]
#
# synth_cat_img = create_u_byte_image_from_vector(synth_cat, height, width, channels)
# io.imshow(synth_cat_img)
# io.show()

# Exercise 24: We could find similar cats by computing the difference between the missing cat
# and all the photos in the databased. Imagine that you only needed to store the 50 weights per cats in
# your database to do the same type of identification? Start by finding the PCA space coordinates of your missing cat:
im_miss = io.imread("data/MissingCatProcessed.jpg")
im_miss_flat = im_miss.flatten()
im_miss_flat = im_miss_flat.reshape(1, -1)
pca_coords = cats_pca.transform(im_miss_flat)
pca_coords = pca_coords.flatten()

# Exercise 25: Plot all the cats in PCA space using the first two dimensions.
# Plot your missing cat in the same plot, with another color and marker.
# Is it placed somewhere sensible and does it have close neighbours?

# plt.scatter(components[:, 0], components[:, 1])
# plt.scatter(pca_coords[0], pca_coords[1], color='r', marker='x')
# plt.xlabel('1st component')
# plt.ylabel('2nd component')
# plt.show()

# Exercise 26: Generate synthetic versions of your cat,
# where you change the n_components_to_use from 1 to for example 50.
n_components_to_use = 50
synth_cat = avg_cat
for idx in range(n_components_to_use):
    w = pca_coords[idx]
    synth_cat = synth_cat + w * cats_pca.components_[idx, :]
synth_cat_img = create_u_byte_image_from_vector(synth_cat, height, width, channels)
io.imshow(synth_cat_img)
io.show()

# Exercise 27: Find the id of the cat that has the smallest and largest distance in PCA space to your missing cat.
# Visualize these cats. Are they as you expected? Do you think your neighours will notice a difference?

# Subtract the missing cat from the data matrix
sub_data = components - pca_coords
# Compute the sum of squared differences
sub_distances = np.linalg.norm(sub_data, axis=1)

# Find the index of the cat that looks most like the missing cat
idx = np.argmin(sub_distances)
# Extract the found cat from the data_matrix
found_cat = data_matrix[idx, :]
# Create an image from the found cat
found_cat_img = create_u_byte_image_from_vector(found_cat, height, width, channels)
# Visualize the found cat
io.imshow(found_cat_img)
io.show()

# Find the index of the cat that looks the least like the missing cat
idx = np.argmax(sub_distances)
# Extract the found cat from the data_matrix
found_cat = data_matrix[idx, :]
# Create an image from the found cat
found_cat_img = create_u_byte_image_from_vector(found_cat, height, width, channels)
# Visualize the found cat
io.imshow(found_cat_img)
io.show()

# Exercise 28: Find the ids of and visualize the 5 closest cats in PCA space. Do they look like your cat?
# You can also find the n closest cats by using the np.argpartition function.
idx = np.argpartition(sub_distances, 5)[:5]
for i in idx:
    found_cat = data_matrix[i, :]
    found_cat_img = create_u_byte_image_from_vector(found_cat, height, width, channels)
    io.imshow(found_cat_img)
    io.show()
