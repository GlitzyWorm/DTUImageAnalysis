from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

### Exercise 1: Read an image ###

# Directory containing data and images
in_dir = "DTUImageAnalysis/exercises/ex1-IntroductionToImageAnalysis/data/"

# X-ray image
im_name = "metacarpals.png"

# Read the image.
# Here the directory and the image name is concatenate
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)

### Exercise 2: Check the image size ###
""" print(im_org.shape) """

### Exercise 3: Check the image data type ###
""" print(im_org.dtype) """

### Exercise 4: Show image ###
""" io.imshow(im_org)
plt.title('Metacarpal image')
io.show() """

### Exercise 5: Display an image using colormap ###
""" io.imshow(im_org, cmap="terrain")
plt.title('Metacarpal image (with colormap)')
io.show() """

### Exercise 7: Try to find a way to automatically scale the visualization, so the pixel with the lowest value in the image is shown as black and the pixel with the highest value in the image is shown as white.  ###
""" vmin = im_org.min()
vmax = im_org.max()

io.imshow(im_org, cmap="gray", vmin=vmin, vmax=vmax)
plt.title('Metacarpal image (with colormap)')
io.show() """

### Exercise 8: Compute the histogram of the image ###
""" plt.hist(im_org.ravel(), bins=256)
plt.title('Image histogram')
io.show()

h = plt.hist(im_org.ravel(), bins=256)

bin_no = 100
count = h[0][bin_no]
print(f"There are {count} pixel values in bin {bin_no}")

bin_left = h[1][bin_no]
bin_right = h[1][bin_no + 1]
print(f"Bin edges: {bin_left} to {bin_right}") """

### Exercise 9: Use the histogram function to find the most common range of intensities ###
# Compute the histogram
""" hist, bins = np.histogram(im_org.ravel(), bins=256)

# Find the index of the bin with the highest count
idx_max = np.argmax(hist)

# The most common range of intensities is then given by the bin edges
range_start = bins[idx_max]
range_end = bins[idx_max + 1]

print(f"The most common range of intensities is from {range_start} to {range_end}") """

### Exercise 10: What is the pixel value at (r, c) = (110, 90) ###
""" r = 110
c = 90
im_val = im_org[r, c]
print(f"The pixel value at (r, c) = ({r}, {c}) is {im_val}") """

### Exercise 11: What does this operation do ###
""" im_org[:30] = 0
io.imshow(im_org)
io.show() """

# Changes the first 30 rows of the image to black

### Exercise 12: Where are the values 1 and where are they 0? ###
""" mask = im_org > 150
io.imshow(mask)
io.show() """

# The values 1 are where the pixel values are greater than 150, and 0 where they are less than 150

### Exercise 13: Wha does this piece of code do? ###
""" im_org[mask] = 255
io.imshow(im_org)
io.show() """

# Changes the pixel values to 255 where the mask is 1 and keeps the original values where the mask is 0

### Exercise 14: Read the image ardeche.jpg and print the image dimensions and its pixel type. How many rows and columns do the image have? ###

# Image
""" im_name_2 = "ardeche.jpg" """

# Read the image.
# Here the directory and the image name is concatenate
# by "+" to give the full path to the image.
""" im_org_2 = io.imread(in_dir + im_name_2)

print(im_org.shape)

print(im_org.dtype) """

# The image has 512 rows and 512 columns

### Exercise 15: What are the (R, G, B) values at (r, c) = (110, 90)? ###
# Assuming im_org_2 is your image
""" r, c = 110, 90  # row and column
pixel_value = im_org_2[r, c]
print(pixel_value) """

### Exercise 16: Try to use NumPy slicing to color the upper half of the photo green ###
# Assuming im_org_2 is your image

# Get the number of rows
""" rows = im_org_2.shape[0] """

# Color the upper half of the image green
""" im_org_2[:int(rows/2), :, 1] = 255
io.imshow(im_org_2)
io.show() """

### Exercise 17: Start by reading your own image and examine the size of it ###
# Image
""" im_name = "star-wars.png" """

# Read the image.
# Here the directory and the image name is concatenate
# by "+" to give the full path to the image.
""" im_org = io.imread(in_dir + im_name)

print(im_org.shape)

print(im_org.dtype) """

### Exercise 18: What is the type of the pixels after rescaling? Try to show the image and inspect the pixel values. Are they still in the range of [0, 255]? ###

""" image_rescaled = rescale(im_org, 0.25, anti_aliasing=True, channel_axis=2) """

# Show rescaled image
""" io.imshow(image_rescaled)
io.show() """

# Check the pixel values
# Convert the image to numpy array
""" img_np = np.array(image_rescaled)

# Check if all pixel values are in the range [0, 255]
is_in_range = np.all((img_np >= 0) & (img_np <= 255))

print(is_in_range) """

### Exercise 19.1: Try to find a way to automatically scale your image so the resulting width (number of columns) is always eaual to 400, no matter the size of the input image? ###

# Find the scaling factor to get the width to 400
""" scale_factor = 400 / im_org.shape[1] """

# Scale the image
""" image_rescaled = rescale(im_org, scale_factor, anti_aliasing=True, channel_axis=2) """

# Show dimensions of the rescaled image
""" print(image_rescaled.shape) """

### Exercise 19.2: Compute and show the histogram of your own image. ###
""" im_gray = color.rgb2gray(im_org)
im_byte = img_as_ubyte(im_gray) """

# Compute the histogram
""" plt.hist(im_org.ravel(), bins=256)
plt.title('Image histogram')
io.show() """

### Exercise 20: Take an image that is very dark and another very light image. Compute and visualize the histograms for the two images. Explain the difference between the two histograms. ###
# Dark image
# Image
""" im_name_dark = "dark-image.jpg" """

# Read the image.
# Here the directory and the image name is concatenate
# by "+" to give the full path to the image.
""" im_dark = io.imread(in_dir + im_name_dark) """

# Compute the histogram
""" plt.hist(im_dark.ravel(), bins=256)
plt.title('Image histogram')
io.show() """

# Light image
# Image
""" im_name_light = "light-image.jpg" """

# Read the image.
# Here the directory and the image name is concatenate
# by "+" to give the full path to the image.
""" im_light = io.imread(in_dir + im_name_light) """

# Compute the histogram
""" plt.hist(im_light.ravel(), bins=256)
plt.title('Image histogram')
io.show() """

# The dark image has a higher peak at the lower end of the histogram, while the light image has a higher peak at the higher end of the histogram

### Exercise 21: Take an image with a bright object on a dark background. Compute and visualise the histograms for the image. Can you recognise the object and the background in the histogram? ###
# I'm using a light image with a dark object on it instead.

# Light image
# Image
""" im_name_light = "light-image.jpg" """

# Read the image.
# Here the directory and the image name is concatenate
# by "+" to give the full path to the image.
""" im_light = io.imread(in_dir + im_name_light) """

# Compute the histogram
""" plt.hist(im_light.ravel(), bins=256)
plt.title('Image histogram')
io.show() """

# The object is the peak at the lower end of the histogram, while the background is the peak at the higher end of the histogram


### Exercise 22: Start by reading and showing the DTUSign1.jpg image ###
# Image
""" im_name = "DTUSign1.jpg" """

# Read the image.
# Here the directory and the image name is concatenate
# by "+" to give the full path to the image.
""" im_org = io.imread(in_dir + im_name) """

# Show the image
""" io.imshow(im_org)
io.show() """

### Exercise 23: Visualize the R, G, and B components individually. Why does the DTU Compute sign look bright on the R channel image and dark on the G and B channels? Why does the walls of the building look bright in all channels? ###

# Visualize the red component of the image
""" r_comp = im_org[:, :, 0]
io.imshow(r_comp)
plt.title('DTU sign image (Red)')
io.show() """

# Visualize the green component of the image
""" g_comp = im_org[:, :, 1]
io.imshow(g_comp)
plt.title('DTU sign image (Green)')
io.show() """

# Visualize the blue component of the image
""" b_comp = im_org[:, :, 2]
io.imshow(b_comp)
plt.title('DTU sign image (Blue)')
io.show() """

# The DTU Compute sign is red, and therefore has a high value in the red channel, but low values in the green and blue channels. The walls of the building are closer to white, and therefore have high values in all channels.

### Exercise 24: Start by reading and showing the DTUSign1.jpg image ###
# Image
""" im_name = "DTUSign1.jpg" """

# Read the image.
# Here the directory and the image name is concatenate
# by "+" to give the full path to the image.
""" im_org = io.imread(in_dir + im_name) """

# Show the image
""" io.imshow(im_org)
io.show() """

### Exercise 25: Show the image again and save it to disk as DTUSign1-marked.jpg using the `io.imsave` function. Try to save the image using different image formats like for example PNG. ###
""" im_org[500:1000, 800:1500, :] = 0 """

# Save the image as a JPG
""" io.imsave(in_dir + "DTUSign1-marked.jpg", im_org) """

# Save the image as a PNG
""" io.imsave(in_dir + "DTUSign1-marked.png", im_org) """

### Exercise 26: Try to create a blue rectangle around the DTU sign and save the resulting image. ###
""" im_org[500:1000, 800:1500, :] = 0
im_org[500:1000, 800:1500, 2] = 255 """

# Save the image as a JPG
""" io.imsave(in_dir + "DTUSign1-blue-marked.jpg", im_org) """


### Exercise 27: Try to automatically create an image based on metacarpals.png where the bones are colored blue. You should use color.gray2rgb and pixel masks. ###
# Image
""" im_name = "metacarpals.png" """

# Read the image.
# Here the directory and the image name is concatenate
# by "+" to give the full path to the image.
""" im_org = io.imread(in_dir + im_name) """

# Convert the image to RGB
""" im_rgb = color.gray2rgb(im_org) """

# Create a mask for the bones
""" mask = im_org > 120 """

# Color the bones blue
""" im_rgb[mask] = [0, 0, 255] """

# Show the image
""" io.imshow(im_rgb)
io.show() """

### Exercise 28: What do you see - can you recognise the inner and outer borders of the bone-shell in the profile ###
# Start by reading the image metacarpals.png

# Image
""" im_name = "metacarpals.png" """

# Read the image.
# Here the directory and the image name is concatenate
# by "+" to give the full path to the image.
""" im_org = io.imread(in_dir + im_name)

p = profile_line(im_org, (342, 77), (320, 160))
plt.plot(p)
plt.ylabel('Intensity')
plt.xlabel('Distance along line')
plt.show() """

# The inner border is the first peak, and the outer border is the second peak

# The image can also be viewed in landscape
""" im_name = "road.png"
im_org = io.imread(in_dir + im_name)
im_gray = color.rgb2gray(im_org)
ll = 200
im_crop = im_gray[40:40 + ll, 150:150 + ll]
xx, yy = np.mgrid[0:im_crop.shape[0], 0:im_crop.shape[1]]
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xx, yy, im_crop, rstride=1, cstride=1, cmap=plt.cm.jet,
                       linewidth=0)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show() """


### Exercise 29: What is the size (number of rows and columns) of the DICOM slice? ###

im_name = "1-442.dcm"
ds = dicom.dcmread(in_dir + im_name)
#print(ds)

# The size of the DICOM slice is 512 x 512

### Exercise 30: Try to find the shape of this image and the pixel type? Does the shape match the size of the image found by inspecting the image header information? ###

im = ds.pixel_array

print(im.shape)
print(im.dtype)

# The shape match the size of the image found by inspecting the image header information

io.imshow(im, vmin=-1000, vmax=1000, cmap='gray')
io.show()
