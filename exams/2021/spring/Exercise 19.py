# Focal length
f = 10

# Distance from camera to fish (in mm)
g = 1100

# Length of fish (in mm)
G = 400

# Size of pixel in the image (in mm) (pixel width divided by CCD chip width)
pixel_mm = 6480 / 5.4

# Calculate the distance to the intersection point (in focus)
# Formula: 1/g + 1/b = 1/f -> b = 1/(1/f - 1/g)
# b = 1 / (1/f - 1/g)
b = f  # For some reason, the formula is simplified to this

# Calculate fish height on CCD
# Formula: b/B = g/G -> B = b * G / g
B = b * G / g

# Calculate the number of pixels the fish occupies
# Formula: B * pixel_mm
n_pixels = B * pixel_mm

# Print the number of pixels
print(n_pixels)
