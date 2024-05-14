### Setup ###
import numpy as np
import pydicom as dicom
from skimage import io, measure, segmentation
from skimage.measure import label, regionprops

in_dir = "data/Aorta/"

### Exercise 3 ###

# Load the dicom image 1-442.dcm
ds = dicom.dcmread(in_dir + "1-442.dcm")

# Load the image AortaROI.png
aorta_roi = io.imread(in_dir + "AortaROI.png")

# Load the image LiverROI.png
liver_roi = io.imread(in_dir + "LiverROI.png")

# Extract the grey-values of the aorta and liver from the dicom image
aorta_values = ds.pixel_array[aorta_roi > 0]
liver_values = ds.pixel_array[liver_roi > 0]

# Compute the mean and standard deviation of the grey-values of the aorta and liver
mean_aorta = aorta_values.mean()
std_aorta = aorta_values.std()
mean_liver = liver_values.mean()
std_liver = liver_values.std()


# Using the gaussian function, find where the two gaussians intersect
def find_threshold(mn_1, std_1, mn_2, std_2):
    return np.roots([1 / std_1 ** 2 - 1 / std_2 ** 2, -2 * mn_1 / std_1 ** 2 + 2 * mn_2 / std_2 ** 2, mn_1 ** 2 / std_1
                     ** 2 - mn_2 ** 2 / std_2 ** 2])


threshold = find_threshold(mean_aorta, std_aorta, mean_liver, std_liver)
print(f"The threshold is: {threshold}")


### Exercise 4 ###

T = 90

# Create a binary mask of the aorta using the threshold
aorta_mask = ds.pixel_array > T

# Remove BLOB's connected to the border
cleaned_aorta = segmentation.clear_border(aorta_mask)

# Perform a BLOB analysis on the aorta mask
label_img = measure.label(cleaned_aorta, connectivity=2)

# Compute the region properties of the aorta mask
regions = regionprops(label_img)

# Ready a single arraylist that can hold the area, perimeter, and circularity of the BLOB's
blob_properties = []

# Compute the circularity of the BLOB's
for region in regions:
    area = region.area
    perimeter = region.perimeter
    if perimeter == 0:
        circularity = 0
    else:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    blob_properties.append([area, perimeter, circularity])

# Only keep BLOB's with a circularity above 0.95 and an area above 200 pixels
blob_properties = np.array(blob_properties)
filtered_blobs = blob_properties[(blob_properties[:, 2] > 0.94) & (blob_properties[:, 0] > 200)]

# If a pixel has a physical side length of 0.75 millimeter, compute the area of the aorta in square millimeters
area_mm2 = np.sum(filtered_blobs[:, 0]) * 0.75 ** 2

print(f"The area of the aorta in square millimeters is: {area_mm2:.2f}")


### Exercise 5 ###

# Print the mean and std of the aoerta values
print(f"Mean aorta: {mean_aorta:.2f}")
print(f"Std aorta: {std_aorta:.2f}")


