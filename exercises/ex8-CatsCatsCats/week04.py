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
# avg_cat = np.mean(data_matrix, axis=0)

# Exercise 4: Visualize the Mean Cat
# Create a mean cat image from the average cat vector
# mean_cat_img = create_u_byte_image_from_vector(avg_cat, height, width, channels)
# io.imshow(mean_cat_img)
# io.show()

# Exercise 6: Use the preprocess_one_cat function to preprocess the photo of the poor missing cat
preprocess_one_cat()

# Exercise 7: Flatten the pixel values of the missing cat so it becomes a vector of values.
# Read the missing cat image
missing_cat = io.imread("data/MissingCatProcessed.jpg")
# Flatten the image
missing_cat_flat = missing_cat.flatten()

# Exercise 8: Subtract you missing cat data from all the rows in the data_matrix and
# for each row compute the sum of squared differences. This can for example be done by:
# sub_data = data_matrix - im_miss_flat
# sub_distances = np.linalg.norm(sub_data, axis=1)

# Subtract the missing cat from the data matrix
sub_data = data_matrix - missing_cat_flat
# Compute the sum of squared differences
sub_distances = np.linalg.norm(sub_data, axis=1)

# Exercise 9: Find the cat that looks most like your missing cat by finding the cat, where the SSD is smallest.
# You can for example use np.argmin.

# Find the index of the cat that looks most like the missing cat
idx = np.argmin(sub_distances)


