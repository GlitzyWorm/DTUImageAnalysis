from skimage import io, color
from skimage.morphology import binary_closing, binary_opening
from skimage.morphology import disk
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage.color import label2rgb
import pydicom as dicom
from scipy.stats import norm
from scipy.spatial import distance


def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap="gray", vmin=-200, vmax=500)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified, cmap="gray")
    ax2.set_title(modified_name)
    ax2.axis('off')
    io.show()


in_dir = "data/"
ct = dicom.read_file(in_dir + 'Training.dcm')
img = ct.pixel_array
print(img.shape)
print(img.dtype)

# Exercise 1: The spleen typically has HU units in the range of 0 to 150.
# Try to make a good visualization of the CT scan  and spleen using (replace the question marks with values):
# io.imshow(img, vmin=-100, vmax=250, cmap='gray')
# io.show()

# Exercise 2: Compute the average and standard deviation of the Hounsfield units
# found in the spleen in the training image.
# Do they correspond to the values found in the above figure?
spleen_roi = io.imread(in_dir + 'SpleenROI.png')
# convert to boolean image
spleen_mask = spleen_roi > 0
spleen_values = img[spleen_mask]

mu_spleen = np.mean(spleen_values)
std_spleen = np.std(spleen_values)

print(f"Average HU: {mu_spleen:.2f}")
print(f"Standard deviation HU: {std_spleen:.2f}")
# Average HU: 49.48
# Standard deviation HU: 15.00

# Exercise 3: Plot a histogram of the pixel values of the spleen. Does it look like they are Gaussian distributed?
# plt.hist(spleen_values, bins=100)
# plt.show()

# Exercise 4: Plot histograms and their fitted Gaussians of several of the tissues types.
# Do they all look like they are Gaussian distributed?

# Spleen
# n, bins, patches = plt.hist(spleen_values, 60, density=1)
# pdf_spleen = norm.pdf(bins, mu_spleen, std_spleen)
# plt.plot(bins, pdf_spleen)
# plt.xlabel('Hounsfield unit')
# plt.ylabel('Frequency')
# plt.title('Spleen values in CT scan')
# plt.show()

# Liver
liver_roi = io.imread(in_dir + 'LiverROI.png')
liver_mask = liver_roi > 0
liver_values = img[liver_mask]
mu_liver = np.mean(liver_values)
std_liver = np.std(liver_values)
# n, bins, patches = plt.hist(liver_values, 60, density=1)
# pdf_liver = norm.pdf(bins, mu_liver, std_liver)
# plt.plot(bins, pdf_liver)
# plt.xlabel('Hounsfield unit')
# plt.ylabel('Frequency')
# plt.title('Liver values in CT scan')
# plt.show()

# Bone
bone_roi = io.imread(in_dir + 'BoneROI.png')
bone_mask = bone_roi > 0
bone_values = img[bone_mask]
mu_bone = np.mean(bone_values)
std_bone = np.std(bone_values)
# n, bins, patches = plt.hist(bone_values, 60, density=1)
# pdf_bone = norm.pdf(bins, mu_bone, std_bone)
# plt.plot(bins, pdf_bone)
# plt.xlabel('Hounsfield unit')
# plt.ylabel('Frequency')
# plt.title('Bone values in CT scan')
# plt.show()

# Fat
fat_roi = io.imread(in_dir + 'FatROI.png')
fat_mask = fat_roi > 0
fat_values = img[fat_mask]
mu_fat = np.mean(fat_values)
std_fat = np.std(fat_values)
# n, bins, patches = plt.hist(fat_values, 60, density=1)
# pdf_fat = norm.pdf(bins, mu_fat, std_fat)
# plt.plot(bins, pdf_fat)
# plt.xlabel('Hounsfield unit')
# plt.ylabel('Frequency')
# plt.title('Fat values in CT scan')
# plt.show()

# Kidney
kidney_roi = io.imread(in_dir + 'KidneyROI.png')
kidney_mask = kidney_roi > 0
kidney_values = img[kidney_mask]
mu_kidney = np.mean(kidney_values)
std_kidney = np.std(kidney_values)
# n, bins, patches = plt.hist(kidney_values, 60, density=1)
# pdf_kidney = norm.pdf(bins, mu_kidney, std_kidney)
# plt.plot(bins, pdf_kidney)
# plt.xlabel('Hounsfield unit')
# plt.ylabel('Frequency')
# plt.title('Kidney values in CT scan')
# plt.show()

# Exercise 5: Plot the fitted Gaussians of bone, fat, kidneys, liver and spleen.
# What classes are easy to seperate and which classes are hard to seperate?

# Hounsfield unit limits of the plot
# min_hu = -200
# max_hu = 1000
# hu_range = np.arange(min_hu, max_hu, 1.0)
# pdf_spleen = norm.pdf(hu_range, mu_spleen, std_spleen)
# pdf_bone = norm.pdf(hu_range, mu_bone, std_bone)
# pdf_fat = norm.pdf(hu_range, mu_fat, std_fat)
# pdf_kidney = norm.pdf(hu_range, mu_kidney, std_kidney)
# pdf_liver = norm.pdf(hu_range, mu_liver, std_liver)
# plt.plot(hu_range, pdf_spleen, 'r--', label="spleen")
# plt.plot(hu_range, pdf_bone, 'g', label="bone")
# plt.plot(hu_range, pdf_fat, 'b', label="fat")
# plt.plot(hu_range, pdf_kidney, 'y', label="kidney")
# plt.plot(hu_range, pdf_liver, 'm', label="liver")
# plt.title("Fitted Gaussians")
# plt.legend()
# plt.show()

# Fat and bone are easy to separate, while liver and kidney are hard to separate.

# Exercise 6: Define the classes that we aim at classifying. Perhaps some classes should be combined into one class?
# The different options are: spleen, liver, kidney, bone, fat.
# The classes spleen and liver are hard to separate, so they should be combined into one class.
# The others are easy to separate and should be kept as separate classes.

# Exercise 7: Compute the class ranges defining fat, soft tissue and bone
# The class ranges are defined by the mean of the Hounsfield units.
# The spleen, liver and kidneys are combined into one class called soft-tissue.

# What is the average of the averages for the spleen, liver and kidneys?
mu_soft_tissue = (mu_spleen + mu_liver + mu_kidney) / 3
std_soft_tissue = (std_spleen + std_liver + std_kidney) / 3

# Ranges
# Soft-tissue: mu_soft_tissue
# Fat: mu_fat
# Bone: mu_bone

# If you have two classes, the threshold between them is defined as the mid-point between the two class value averages.
# Thresholds
t_fat_soft = (mu_soft_tissue + mu_fat) / 2
t_soft_bone = (mu_fat + mu_bone) / 2
t_background = -200

# Exercise 8: Create class images: fat_img, soft_img and bone_img representing the fat, soft tissue
# and bone found in the image.
fat_img = (img > t_background) & (img <= t_fat_soft)
soft_img = (img > t_fat_soft) & (img <= t_soft_bone)
bone_img = img > t_soft_bone

# Exercise 9: Visualize your classification result and compare it to the anatomical image in the start of the exercise.
# Do your results look plausible?

# Show images
# show_comparison(img, fat_img, 'Fat')
# show_comparison(img, soft_img, 'Soft tissue')
# show_comparison(img, bone_img, 'Bone')

# label_img = fat_img + 2 * soft_img + 3 * bone_img
# image_label_overlay = label2rgb(label_img)
# show_comparison(img, image_label_overlay, 'Classification result')

# Exercise 9: Plot the fitted Gaussians of the training values and manually find the intersection between the curves.
# Use the intersection points to define the thresholds between the classes.


min_hu = -200
max_hu = 1000
hu_range = np.arange(min_hu, max_hu, 1.0)
pdf_soft_tissue = norm.pdf(hu_range, mu_soft_tissue, std_soft_tissue)
pdf_bone = norm.pdf(hu_range, mu_bone, std_bone)
pdf_fat = norm.pdf(hu_range, mu_fat, std_fat)
# plt.plot(hu_range, pdf_soft_tissue, 'r--', label="soft-tissue")
# plt.plot(hu_range, pdf_bone, 'g', label="bone")
# plt.plot(hu_range, pdf_fat, 'b', label="fat")
# plt.title("Fitted Gaussians")
# plt.legend()
# plt.show()

# Fat and soft-tissue = -50
# Soft-tissue and bone = 250
# Background = -200

# Exercise 10: Use the same technique as in exercise 7, 8 and 9 to visualize your classification results.
# Did it change compared to the minimum distance classifier?
t_fat_soft = -50
t_soft_bone = 250
t_background = -200

# Exercise 8: Create class images: fat_img, soft_img and bone_img representing the fat, soft tissue
# and bone found in the image.
fat_img_2 = (img > t_background) & (img <= t_fat_soft)
soft_img_2 = (img > t_fat_soft) & (img <= t_soft_bone)
bone_img_2 = img > t_soft_bone

label_img = fat_img_2 + 2 * soft_img_2 + 3 * bone_img_2
image_label_overlay = label2rgb(label_img)
# show_comparison(img, image_label_overlay, 'Classification result')


# Exercise 11: Use norm.pdf to find the optimal class ranges between fat, soft tissue and bone.

# Define the classify function
def classify(test_value):
    if norm.pdf(test_value, mu_fat, std_fat) > norm.pdf(test_value, mu_soft_tissue, std_soft_tissue):
        # The value is more likely to be fat
        return 'fat'
    elif norm.pdf(test_value, mu_soft_tissue, std_soft_tissue) > norm.pdf(test_value, mu_bone, std_bone):
        # The value is more likely to be soft tissue
        return 'soft tissue'
    else:
        # The value is more likely to be bone
        return 'bone'


# Create a lookup table for all possible HU values
lookup_table = {}
# for i in range(-1000, 3000):
    # lookup_table[i] = classify(i)

# Find the HU values where the most probable class changes
# class_changes = {hu - 1: lookup_table[hu - 1] for hu in range(-999, 3000) if lookup_table[hu] != lookup_table[hu - 1]}

# print(class_changes)

# Exercise 12
t_1 = 20
t_2 = 80

spleen_estimate = (img > t_1) & (img < t_2)
spleen_label_colour = color.label2rgb(spleen_estimate)
io.imshow(spleen_label_colour)
plt.title("First spleen estimate")
io.show()

# Exercise 12: Use the above morphological operations to seperate the spleen from other organs and close holes.
# Change the values where there are question marks to change the size of the used structuring elements.
# footprint = disk(5)
# closed = binary_closing(spleen_estimate, footprint)

# footprint = disk(5)
# opened = binary_opening(closed, footprint)

# spleen_label_colour = color.label2rgb(opened)
# io.imshow(spleen_label_colour)
# plt.title("Second spleen estimate")
# io.show()


