### Setup ###
from skimage import io
from skimage.color import rgb2gray

in_dir = "data/ChangeDetection/"

### Question 13 ###

# Load the images background.png and new_frame.png
background = io.imread(in_dir + "background.png")
new_frame = io.imread(in_dir + "new_frame.png")

# Convert the images to grayscale
background_gray = rgb2gray(background)
new_frame_gray = rgb2gray(new_frame)

# Update the background using the new frame
alpha = 0.90
new_background = alpha * background_gray + (1 - alpha) * new_frame_gray

# Compute the absolute difference between the new background and the new frame
abs_diff = abs(new_background - new_frame_gray)

# Compute how many pixels in the difference image are greater than 0.1
changed_pixels = (abs_diff > 0.1).sum()

# Print the number of changed pixels
print(changed_pixels)

# Compute the average value of the new background in the region [150:200, 150:200]
avg_value = new_background[150:200, 150:200].mean()

# Print the average value
print(avg_value)