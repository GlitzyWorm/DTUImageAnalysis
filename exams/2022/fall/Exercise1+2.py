### Setup ###
import numpy as np
import pydicom as dicom
from skimage import io

in_dir = "data/dicom/"

### Exercise 1 ###

# Load DICOM image (1-162.dcm) from in_dir
ds = dicom.dcmread(in_dir + "1-162.dcm")

# Load annotated regions from BackROI.png, LiverROI.png, KidneyROI.png and AortaROI.png
back_roi = io.imread(in_dir + "BackROI.png")
liver_roi = io.imread(in_dir + "LiverROI.png")
kidney_roi = io.imread(in_dir + "KidneyROI.png")
aorta_roi = io.imread(in_dir + "AortaROI.png")

# Convert to binary masks
back_roi = back_roi > 0
liver_roi = liver_roi > 0
kidney_roi = kidney_roi > 0
aorta_roi = aorta_roi > 0

# Create values for the liver, kidney and aorta
img = ds.pixel_array
back_values = img[back_roi]
liver_values = img[liver_roi]
kidney_values = img[kidney_roi]
aorta_values = img[aorta_roi]

mu_back = np.mean(back_values)
std_back = np.std(back_values)

mu_liver = np.mean(liver_values)
std_liver = np.std(liver_values)

mu_kidney = np.mean(kidney_values)
std_kidney = np.std(kidney_values)

mu_aorta = np.mean(aorta_values)
std_aorta = np.std(aorta_values)

# Thresholds
t_background = mu_back

# Threshold between liver and kidney (t_liver_kidney)
t1 = (mu_liver + mu_kidney) / 2

# Threshold between kidney and aorta (t_kidney_aorta)
t2 = (mu_kidney + mu_aorta) / 2

# Segment the ds image using the thresholds. All values greater than t1 and less than t2 should be set to 1, otherwise 0
segmented = np.zeros_like(img)
segmented[(img > t1) & (img < t2)] = 1

# Computes the DICE score between the resulting segmented image and the KidneyROI.png image.
dice = 2 * np.sum(segmented * kidney_roi) / (np.sum(segmented) + np.sum(kidney_roi))

# Print the thresholds
print(f"Threshold background: {t_background}")
print(f"Threshold liver-kidney: {t1}")
print(f"Threshold kidney-aorta: {t2}")

### Exercise 2 ###

# Print the DICE score
print(f"DICE score: {dice}")



