### Setup ###
import numpy as np
import pydicom as dicom
from skimage import io, measure
from skimage.morphology import closing, disk, opening

in_dir = "data/"


### Q. 14 ###

# Load the DICOM file 1-179.dcm and store it in the variable ds.
ds = dicom.dcmread(in_dir + "1-179.dcm")

# Load the image LiverROI.png and store it in the variable mask.
mask = io.imread(in_dir + "LiverROI.png")

# Extract the values of the pixels in the mask from the DICOM file and store them in the variable mask_values.
mask_values = ds.pixel_array[mask]

# Compute the average value and the standard deviation of the pixel values in the mask
# and store them in the variables avg and std, respectively.
avg = mask_values.mean()
std = mask_values.std()

# Define a lower threshold, T1, as the average value minus the standard deviation,
# and an upper threshold, T2, as the average value plus the standard deviation.
T1 = avg - std
T2 = avg + std

# Segment ds where all pixels with values larger than T1 and smaller than T2 are set to 1, and the rest to 0.
segmented = (ds.pixel_array > T1) & (ds.pixel_array < T2)

# Count the number of pixels with value 1 in the segmented image and store it in the variable n_pixels.
n_pixels = np.sum(segmented)

# Print the number of pixels
print(n_pixels)


### Q. 15 ###

# Set the new thresholds
T1 = 90
T2 = 140

# Segment ds where all pixels with values larger than T1 and smaller than T2 are set to 1, and the rest to 0.
segmented = (ds.pixel_array > T1) & (ds.pixel_array < T2)

# Apply a morphological closing using a disk-shaped structuring element with a radius of 3 pixels.
footprint = disk(3)
segmented_closed = closing(segmented, footprint)

# Apply a morphological opening using a disk-shaped structuring element with a radius of 3 pixels.
segmented_opened = opening(segmented_closed, footprint)

# Do a BLOB analysis with 8-connectivity and store the number of BLOBs in the variable n_blobs.
label_img = measure.label(segmented_opened, connectivity=2)

# Compute the area of all BLOBs
props = measure.regionprops_table(label_img, properties=('label', 'area'))

# Find the largest BLOB in terms of area and store its area in the variable largest_blob_area.
largest_blob_area = np.max(props['area'])

# Print the area of the largest BLOB
print(largest_blob_area)