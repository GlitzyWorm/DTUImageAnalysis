### Setup ###
from skimage import io, measure
from skimage.color import rgb2hsv
from skimage.morphology import disk, erosion, dilation

in_dir = "data/CarData/"

### Exercise 8 ###

# Load the image car.png
car_img = io.imread(in_dir + "car.png")

# Convert from RGB to HSV
car_hsv = rgb2hsv(car_img)

# Extract the saturation channel
car_sat = car_hsv[:, :, 1]

# Create a mask for the car
car_mask = car_sat > 0.7

# Perform a morphological erosion using a disk of radius 6
footprint = disk(6)
car_mask_eroded = erosion(car_mask, footprint)

# Perform a morphological dilation using a disk of radius 4
footprint = disk(4)
car_mask_dilated = dilation(car_mask_eroded, footprint)

# Count the number of pixels in the mask with the value True
count = car_mask_dilated.sum()
print(f"Number of pixels in the mask: {count}")


### Exercise 9 ###

# Load the image road.png
road_img = io.imread(in_dir + "road.png")

# Convert from RGB to HSV
road_hsv = rgb2hsv(road_img)

# Extract the value channel
road_val = road_hsv[:, :, 2]

# Create a mask for the road
road_mask = road_val > 0.9

# Find all BLOB's using 8-connectivity
label_img = measure.label(road_mask, connectivity=2)

# Compute the area of each BLOB
props = measure.regionprops(label_img)

# Display road_mask
io.imshow(road_mask)
io.show()

# Find the two largest areas
areas = [prop.area for prop in props]
areas.sort(reverse=True)
print(f"Two largest areas: {areas[0:2]}")

# Filter out all the BLOB's with an area less than the second largest area
filtered_label_img = label_img.copy()
for prop in props:
    if prop.area < areas[1]:
        filtered_label_img[filtered_label_img == prop.label] = 0

# Display filtered_label_img
io.imshow(filtered_label_img)
io.show()
