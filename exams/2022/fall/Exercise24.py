### Setup ###
from skimage import io, color

in_dir = "data/ChangeDetection/"

### Exercise 24 ###

# Load the images
change1 = io.imread(in_dir + "change1.png")
change2 = io.imread(in_dir + "change2.png")

# Convert the images to grayscale
change1_gray = color.rgb2gray(change1)
change2_gray = color.rgb2gray(change2)

# Compute the absolute difference between the two images
diff = abs(change1_gray - change2_gray)

# Compute how many pixels in the difference image are greater than 0.3. Call these changed_pixels
changed_pixels = (diff > 0.3).sum()

# Compute the percentage of changed pixels compared to the total number of pixels in the images
total_pixels = diff.size
percentage_changed = changed_pixels / total_pixels * 100

print(f"The percentage of changed pixels is: {percentage_changed:.2f}%")
