### Setup ###
import numpy as np
import pydicom
from matplotlib import pyplot as plt
from scipy.stats import norm
from skimage import io
from skimage.morphology import disk, dilation, erosion
from skimage.measure import label, regionprops

in_dir = "data/Abdominal/"

### Question 1 ###

in_dir = "data/Abdominal/"
im_name = "1-166.dcm"

ds = pydicom.dcmread(in_dir + im_name)
img = ds.pixel_array

### Answer to Question 1 ###

kidney_l_roi = io.imread(in_dir + 'kidneyROI_l.png')
kidney_l_mask = kidney_l_roi > 0
kidney_l_values = img[kidney_l_mask]
(mu_kidney_l, std_kidney_l) = norm.fit(kidney_l_values)
print(f"Answer: kidney_l: Average {mu_kidney_l:.0f} standard deviation {std_kidney_l:.0f}")

kidney_r_roi = io.imread(in_dir + 'kidneyROI_r.png')
kidney_r_mask = kidney_r_roi > 0
kidney_r_values = img[kidney_r_mask]
(mu_kidney_r, std_kidney_r) = norm.fit(kidney_r_values)
print(f"Answer: kidney_r: Average {mu_kidney_r:.0f} standard deviation {std_kidney_r:.0f}")

# Load the image LiverROI.png
liver_roi = io.imread(in_dir + "LiverROI.png")

# Compute the average and standard deviation of the Hounsfield units for the liver
liver_hu = ds.pixel_array * liver_roi
liver_hu_values = liver_hu[liver_hu > 0]

# Compute the average Hounsfield unit value for the liver
liver_hu_mean = liver_hu_values.mean()

# Compute the standard deviation of the Hounsfield unit values for the liver
liver_hu_std = liver_hu_values.std()

# Print
print(f"Average HU for liver: {liver_hu_mean}")
print(f"Standard deviation of HU for liver: {liver_hu_std}")

# Compute threshold t_1 that is the average liver HU minus the standard deviation of liver HU
t_1 = liver_hu_mean - liver_hu_std

# Compute threshold t_2 that is the average liver HU plus the standard deviation of liver HU
t_2 = liver_hu_mean + liver_hu_std

### Answer to Question 2 ###

print(f"Threshold 1: {t_1}\n" 
      f"Threshold 2: {t_2}")

# Create a binary mask for ds that is 1 where the HU values are between t_1 and t_2, and 0 elsewhere
binary_mask = ((ds.pixel_array >= t_1) & (ds.pixel_array <= t_2)).astype(int)

# Dilate the binary mask using a disk with a radius of 3 pixels
dilated_mask = dilation(binary_mask, disk(3))

# Erode the dilated mask using a disk with a radius of 10 pixels
eroded_mask = erosion(dilated_mask, disk(10))

# Dilate the eroded mask using a disk with a radius of 10 pixels
final_mask = dilation(eroded_mask, disk(10))

# Find all BLOB's in the final mask
labelled_mask = label(final_mask, connectivity=2)

# Compute the region properties of the labelled mask
regions = regionprops(labelled_mask)

# Remove BLOB's with an area less than 1500 pixels or an area greater than 7000 pixels or a perimeter less than 300
filtered_region = None
for region in regions:
    if 1500 <= region.area <= 7000 and region.perimeter >= 300:
        filtered_regions = region


# Extract the filtered region from the labelled mask
filtered_regions = labelled_mask == filtered_regions.label

# Compute the DICE score between the filtered regions and the liver ROI
dice = 2 * np.sum(filtered_regions * liver_roi) / (np.sum(filtered_regions) + np.sum(liver_roi))

### Answer to Question 3 ###

# Print
print(f"DICE score between filtered region and liver ROI: {dice}")

