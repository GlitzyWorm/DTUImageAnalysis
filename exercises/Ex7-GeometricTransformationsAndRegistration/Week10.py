import matplotlib.pyplot as plt
import math

import numpy as np
from skimage import io, img_as_float
from skimage.transform import rotate, matrix_transform
from skimage.transform import EuclideanTransform
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import swirl


def show_comparison(original, transformed, transformed_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(transformed)
    ax2.set_title(transformed_name)
    ax2.axis('off')
    io.show()


in_dir = "data/"

# Exercise 1: Rotate image

# Read NusaPenida.png and call it im_org
# im_org = io.imread(in_dir + "NusaPenida.png")

# angle in degrees - counterclockwise
# rotation_angle = 10
# rot_center = [0, 10]
# rotated_img = rotate(im_org, rotation_angle, center=rot_center)
# show_comparison(im_org, rotated_img, "Rotated image")

# Exercise 2: Experiment with different rotations and centers
# rotation_angle = 10
# rot_center = [0, 10]
# rotated_img = rotate(im_org, rotation_angle, center=rot_center)
# show_comparison(im_org, rotated_img, "Rotated image: 10")
#
# rotation_angle = 45
# rot_center = [0, 10]
# rotated_img = rotate(im_org, rotation_angle, center=rot_center)
# show_comparison(im_org, rotated_img, "Rotated image: 45")
#
# rotation_angle = 90
# rot_center = [0, 10]
# rotated_img = rotate(im_org, rotation_angle, center=rot_center)
# show_comparison(im_org, rotated_img, "Rotated image: 90")

# Exercise 3: Try the rotation with background filling mode reflect and wrap and notice the results and differences.
# rotation_angle = 10
# rot_center = [0, 0]
# rotated_img = rotate(im_org, rotation_angle, center=rot_center, mode='reflect')
# show_comparison(im_org, rotated_img, "Rotated image: reflect")
#
# rotation_angle = 10
# rot_center = [0, 0]
# rotated_img = rotate(im_org, rotation_angle, center=rot_center, mode='wrap')
# show_comparison(im_org, rotated_img, "Rotated image: wrap")

# Exercise 4: Try the rotation with background filling mode constant and notice the results and differences.
# rotation_angle = 10
# rot_center = [0, 0]
# rotated_img = rotate(im_org, rotation_angle, center=rot_center, mode='constant', cval=1)
# show_comparison(im_org, rotated_img, "Rotated image: constant")
#
# rotated_img = rotate(im_org, rotation_angle, center=rot_center, mode='constant', cval=0)
# show_comparison(im_org, rotated_img, "Rotated image: constant")
#
# rotated_img = rotate(im_org, rotation_angle, center=rot_center, mode='constant', cval=0.5)
# show_comparison(im_org, rotated_img, "Rotated image: constant")

# Exercise 5: Rotate image using automatic resizing
# rotation_angle = 45
# rot_center = [0, 0]
# rotated_img = rotate(im_org, rotation_angle, mode='reflect', center=rot_center, resize=True)
# show_comparison(im_org, rotated_img, "Rotated image: resize")

# Exercise 6: Euclidean transformation

# angle in radians - counterclockwise
# rotation_angle = 10.0 * math.pi / 180.
# trans = [10, 20]
# tform = EuclideanTransform(rotation=rotation_angle, translation=trans)
# print(tform.params)

# Exercise 7: Apply the computed transformation to the image using the warp function
# transformed_img = warp(im_org, tform)
# show_comparison(im_org, transformed_img, "Transformed image")

# Exercise 8: Inverse transformation
# transformed_img = warp(im_org, tform.inverse)
# show_comparison(im_org, transformed_img, "Inverse transformed image")

# Only with rotation
# rotation_angle = 45.0 * math.pi / 180
# tform = EuclideanTransform(rotation=rotation_angle)
# transformed_img = warp(im_org, tform)
# transformed_inv_img = warp(transformed_img, tform.inverse)
# show_comparison(transformed_img, transformed_inv_img, "Inverse transformed image")

# Exercise 9: Similarity transformation
# Define a SimilarityTransform with an angle of 15 degrees,
# a translation of (40, 30) and a scaling of 0.6 and test it on the image.

# rotation_angle = 15.0 * math.pi / 180
# trans = [40, 30]
# scale = 0.6
# tform = SimilarityTransform(rotation=rotation_angle, translation=trans, scale=scale)
# print(tform.params)
# transformed_img = warp(im_org, tform)
# show_comparison(im_org, transformed_img, "Transformed image")

# Exercise 10: Swirl transformation
# str = 10
# rad = 300
# swirl_img = swirl(im_org, strength=str, radius=rad)
# show_comparison(im_org, swirl_img, "Swirl image")
#
# str = 10
# rad = 300
# c = [500, 400]
# swirl_img = swirl(im_org, strength=str, radius=rad, center=c)
# show_comparison(im_org, swirl_img, "Swirl image with moved center")

### Landmark based registration ###

# Exercise 11: Load the images
# in_dir = "exercises/Ex7-GeometricTransformationsAndRegistration/data/"
src_img = io.imread(in_dir + "Hand1.jpg")
dst_img = io.imread(in_dir + "Hand2.jpg")
# blend = 0.5 * img_as_float(src_img) + 0.5 * img_as_float(dst_img)
# io.imshow(blend)
# io.show()

# Exercise 12: Define the manual landmarks
src = np.array([[588, 274], [328, 179], [134, 398], [260, 525], [613, 448]])

# plt.imshow(src_img)
# plt.plot(src[:, 0], src[:, 1], '.r', markersize=12)
# plt.show()

# Exercise 13: Define the landmarks

# First show the image
# io.imshow(dst_img)
# io.show()

# Define the landmarks
dst = np.array([[594, 444], [276, 438], [198, 270], [383, 160], [630, 300]])

# plt.imshow(dst_img)
# plt.plot(dst[:, 0], dst[:, 1], '.r', markersize=12)
# plt.show()

# Plot landmarks to verify
# fig, ax = plt.subplots()
# ax.plot(src[:, 0], src[:, 1], '-r', markersize=12, label="Source")
# ax.plot(dst[:, 0], dst[:, 1], '-g', markersize=12, label="Destination")
# ax.invert_yaxis()
# ax.legend()
# ax.set_title("Landmarks before alignment")
# plt.show()

# Exercise 14: Compute the objective function. I.e., how well-aligned the two hands are
e_x = src[:, 0] - dst[:, 0]
error_x = np.dot(e_x, e_x)
e_y = src[:, 1] - dst[:, 1]
error_y = np.dot(e_y, e_y)
f = error_x + error_y
print(f"Landmark alignment 'before' error F: {f}")

# Exercise 15:Visualize the transformed source landmarks together with the destination landmarks.
# Also compute the objective function F using the transformed points. What do you observe?
tform = EuclideanTransform()
tform.estimate(src, dst)
src_transform = matrix_transform(src, tform.params)

# fig, ax = plt.subplots()
# ax.plot(src_transform[:, 0], src_transform[:, 1], '-b', markersize=12, label="Transformed source")
# ax.plot(dst[:, 0], dst[:, 1], '-g', markersize=12, label="Destination")
# ax.invert_yaxis()
# ax.legend()
# ax.set_title("Landmarks after alignment")
# plt.show()
#
# e_x = src_transform[:, 0] - dst[:, 0]
# error_x = np.dot(e_x, e_x)
# e_y = src_transform[:, 1] - dst[:, 1]
# error_y = np.dot(e_y, e_y)
# f = error_x + error_y
# print(f"Landmark alignment 'after' error F: {f}")

# Exercise 16: Apply the estimated transformation to the source image and visualize the result.
# transformed_img = warp(src_img, tform.inverse)
# show_comparison(src_img, transformed_img, "Transformed image")

# also try to blend the warped image destination image like in exercise 11
# blend = 0.5 * img_as_float(transformed_img) + 0.5 * img_as_float(dst_img)
# io.imshow(blend)
# io.show()