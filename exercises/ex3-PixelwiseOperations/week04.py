import cv2
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu

# Exercise 1: Start by reading the image and inspect the histogram. Is it a bimodal histogram? Do you think it
# will be possible to segment it so only the bones are visible?
# The image is vertebra.png

# Pycharm
in_dir = "data/"

# X-ray image
im_name = "vertebra.png"

im_org = io.imread(in_dir + im_name)


# plt.hist(im_org.ravel(), bins=256)
# plt.title('Image histogram')
# io.show()
# The image is bimoal, so it is possible to segment it so only the bones are visible.

# Exercise 2: Compute the minimum and maximum values of the image. Is the full scale of the gray-scale spectrum used
# or can we enhance the appearance of the image?

# Minimum and maximum values of the image
# min_val = im_org.min()
# max_val = im_org.max()
# print("Minimum value: ", min_val)
# print("Maximum value: ", max_val)
# The full scale of the gray-scale spectrum is not used, so we can enhance the appearance of the image.

# Exercise 3: Use img_as_float to compute a new float version of your input image. Compute the minimum and maximum
# values of this float image. Can you verify that the float image is equal to the original image,
# where each pixel value is divided by 255?

# Float version of the input image
# im_float = img_as_float(im_org)


# Minimum and maximum values of the float image
# min_val_float = im_float.min()
# max_val_float = im_float.max()
# print("Minimum value of the float image: ", min_val_float)
# print("Maximum value of the float image: ", max_val_float)

# Verify that the float image is equal to the original image, where each pixel value is divided by 255
# im_float_min = min_val / 255
# im_float_max = max_val / 255
# print("Are the float images equal? ", round(im_float_min, 2) == round(min_val_float, 2)
#       and round(im_float_max, 2) == round(max_val_float, 2))

# Exercise 4: Use img_as_ubyte on the float image you computed in the previous exercise.
# Compute the minimum and maximum values of this image. Are they as expected?

# Ubyte version of the float image
# im_ubyte = img_as_ubyte(im_float)

# Minimum and maximum values of the ubyte image
# min_val_ubyte = im_ubyte.min()
# max_val_ubyte = im_ubyte.max()
# print("Minimum value of the ubyte image: ", min_val_ubyte)
# print("Maximum value of the ubyte image: ", max_val_ubyte)


# They are as expected, the minimum value is 57 and the maximum value is 235.

# Exercise 5:  Implement a Python function called histogram_stretch. It can, for example, follow this example:

# def histogram_stretch(img_in):
#     """
#     Stretches the histogram of an image
#     :param img_in: Input image
#     :return: Image, where the histogram is stretched so the min values is 0 and the maximum value 255
#     """
#     # img_as_float will divide all pixel values with 255.0
#     img_float = img_as_float(img_in)
#     min_val = img_float.min()
#     max_val = img_float.max()
#     min_desired = 0.0
#     max_desired = 1.0
#
#     # Do something here
#     img_out = (max_desired - min_desired) / (max_val - min_val) * (img_float - min_val) + min_desired
#
#     # img_as_ubyte will multiply all pixel values with 255.0 before converting to unsigned byte
#     return img_as_ubyte(img_out)


# Exercise 6: Test your histogram_stretch on the vertebra.png image. Show the image before and
# after the histogram stretching. What changes do you notice in the image? Are the important structures more visible?

# Stretch the histogram of the image
# im_stretched = histogram_stretch(im_org)
# io.imshow(im_stretched)
# plt.title("Stretched image")
# io.show()
#
# io.imshow(im_org)
# plt.title("Original image")
# io.show()


# The important structures are more visible, and the image has a better contrast.

# Exercise 7: Implement a function, gamma_map(img, gamma), that:
# 1. Converts the input image to float
# 2. Do the gamma mapping on the pixel values using $$g(x,y) = f(x,y)^\gamma \enspace .$$ with numpy power
# 3. Returns the resulting image as an unsigned byte image.

# Gamma mapping function
# def gamma_map(img, gamma):
#     """
#     Gamma mapping of an image
#     :param img: Input image
#     :param gamma: Gamma value
#     :return: Image, where the gamma mapping is applied
#     """
#     # img_as_float will divide all pixel values with 255.0
#     img_float = img_as_float(img)
#
#     # Do something here
#     img_out = np.power(img_float, gamma)
#     # img_out = img_float ** gamma
#
#     # img_as_ubyte will multiply all pixel values with 255.0 before converting to unsigned byte
#     return img_as_ubyte(img_out)


# Exercise 8: Test your gamma_map function on the vertebra image or another image of your choice.
# Try different values of $\gamma$, for example 0.5 and 2.0.
# Show the resulting image together with the input image.
# Can you see the differences in the images?

# Gamma mapping of the image
# gamma_val = 2.0
# im_gamma = gamma_map(im_org, gamma_val)

# io.imshow(im_org)
# plt.title("Original image")
# io.show()
#
# io.imshow(im_gamma)
# plt.title("Gamma image")
# io.show()

# The higher the gamma value, the darker the image becomes. The lower the gamma value, the brighter the image becomes.

# Exercise 9: Implement a function, threshold_image:
# def threshold_image(img_in, thres):
#     """
#     Apply a threshold in an image and return the resulting image
#     :param img_in: Input image
#     :param thres: The treshold value in the range [0, 255]
#     :return: Resulting image (unsigned byte) where background is 0 and foreground is 255
#     """
#
#     # Do something here
#     img_out = img_in > thres
#     return img_as_ubyte(img_out)


# Exercise 10: Test your threshold_image function on the vertebra image with different thresholds.
# It is probably not possible to find a threshold that seperates the bones from the background,
# but can you find a threshold that seperates the human from the background?

# Threshold the image
# threshold_val = 150
# im_thresh = threshold_image(im_org, threshold_val)
#
# io.imshow(im_thresh)
# plt.title("Threshold image")
# io.show()

# The threshold value 150 separates the human from the background.

# Exercise 11: Read the documentation of Otsu's method
# and use it to compute and apply a threshold to the vertebra image.

# # Compute the threshold
# thresh = threshold_otsu(im_org)
#
# # Print the threshold
# print("Threshold value: ", thresh)
#
# # Apply the threshold
# im_thresh = im_org > thresh
#
# io.imshow(im_thresh)
# plt.title("Threshold image")
# io.show()

# The threshold value is 148, and it is very close to the threshold value 150 we found in the previous exercise.

# Exercicse 12: Use your camera to take some pictures of yourself or a friend.
# Try to take a picture on a dark background.
# Convert the image to grayscale and try to find a threshold that creates a silhouette image
# (an image where the head is all white and the background black).

# Use the camera to take a picture of yourself or a friend
# capture_from_camera_and_show_images()

# def capture_from_camera_and_show_images():
#     print("Starting image capture")
#
#     print("Opening connection to camera")
#     url = 0
#     use_droid_cam = False
#     if use_droid_cam:
#         url = "http://192.168.1.120:4747/video"
#     cap = cv2.VideoCapture(url)
#     if not cap.isOpened():
#         print("Cannot open camera")
#         exit()
#
#     # Get image
#     ret, frame = cap.read()
#     if not ret:
#         print("Can't receive frame. Exiting ...")
#         exit()
#
#     # Convert the image to grayscale
#     new_image_gray = color.rgb2gray(frame)
#
#     # Find a threshold that creates a silhouette image
#     thresh = threshold_otsu(new_image_gray)
#
#     # Print the threshold
#     print("Threshold value: ", thresh)
#
#     # Apply the threshold
#     im_thresh = new_image_gray > thresh
#
#     io.imshow(im_thresh)
#     plt.title("Threshold image")
#     io.show()
#
#
# capture_from_camera_and_show_images()

# Exercise 13: Create a function detect_dtu_signs that takes as input a color image and returns an image,
# where the blue sign is identified by foreground pixels

# def detect_dtu_signs(img, red, blue):
#     """
#     Detects the DTU sign in an image
#     :param img: Input color image.
#     :param red: Boolean, check for red
#     :param blue: Boolean, check for blue
#     :return: image where the blue sign is identified by foreground pixels.
#     """
#     r_comp = img[:, :, 0]
#     g_comp = img[:, :, 1]
#     b_comp = img[:, :, 2]
#
#     # Check if more than one color is checked
#     if red and blue:
#         raise ValueError("Only one color can be checked")
#
#     # Check which color to detect
#     if red:
#         segm_red = (r_comp > 150) & (r_comp < 220) & (g_comp < 60) & (b_comp < 70)
#         return segm_red
#
#     if blue:
#         segm_blue = (r_comp < 10) & (g_comp > 85) & (g_comp < 105) & (b_comp > 180) & (b_comp < 200)
#         return segm_blue
#
#     # # Threshold the image
#     # segm_blue = (r_comp < 10) & (g_comp > 85) & (g_comp < 105) & \
#     #             (b_comp > 180) & (b_comp < 200)
#     #
#     # return segm_blue
#
#
# sign_image = io.imread(in_dir + "DTUSigns2.jpg")
# # io.imshow(sign_image)
# # plt.title("Original image")
# # io.show()
#
# signs = detect_dtu_signs(sign_image, True, False)
# io.imshow(signs)
# plt.title("Sign image")
# io.show()

# Exercise 14.1:

# sign_image = io.imread(in_dir + "DTUSigns2.jpg")
# hsv_img = color.rgb2hsv(sign_image)
# hue_img = hsv_img[:, :, 0]
# value_img = hsv_img[:, :, 2]
# fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(8, 2))
# ax0.imshow(sign_image)
# ax0.set_title("RGB image")
# ax0.axis('off')
# ax1.imshow(hue_img, cmap='hsv')
# ax1.set_title("Hue channel")
# ax1.axis('off')
# ax2.imshow(value_img)
# ax2.set_title("Value channel")
# ax2.axis('off')
#
# fig.tight_layout()
# io.show()

# Exercise 15: Now make a sign segmentation function using tresholding in HSV space
# and locate both the blue and the red sign.

def detect_dtu_signs_hsv(img, red, blue):
    """
    Detects the DTU sign in an image
    :param img: Input color image.
    :param red: Boolean, check for red
    :param blue: Boolean, check for blue
    :return: image where the blue sign is identified by foreground pixels.
    """
    hsv_img = color.rgb2hsv(img)
    hue_img = hsv_img[:, :, 0]
    value_img = hsv_img[:, :, 2]

    # Check if more than one color is checked
    if red and blue:
        raise ValueError("Only one color can be checked")

    # Check which color to detect
    if red:
        segm_red = (hue_img > 0.7) | (hue_img < 0.0) & (value_img > 0.9)
        return segm_red

    if blue:
        segm_blue = (hue_img > 0.55) & (hue_img < 0.6) & (value_img > 0.3)
        return segm_blue

# Test the function
sign_image = io.imread(in_dir + "DTUSigns2.jpg")
red_sign = detect_dtu_signs_hsv(sign_image, True, False)
blue_sign = detect_dtu_signs_hsv(sign_image, False, True)
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(8, 2))
ax0.imshow(sign_image)
ax0.set_title("Original image")
ax0.axis('off')
ax1.imshow(red_sign, cmap='gray')
ax1.set_title("Red sign")
ax1.axis('off')
ax2.imshow(blue_sign, cmap='gray')
ax2.set_title("Blue sign")
ax2.axis('off')

fig.tight_layout()
io.show()
