import math

### Exercise 1 - Calculate the angle of a right-angled triangle

# a = 10
# b = 3
#
# theta_rad = math.atan2(a, b)
# theta_deg = math.degrees(theta_rad)
#
# print(f"The angle of the right-angled triangle is {theta_deg:.2f} degrees")

### Exercise 2 - Calculate the distance from the lens to where the rays are focused (b) (where the CCD should be placed)
def camera_b_distance(f, g):
    """
    camera_b_distance returns the distance (b) where the CCD should be placed
    when the object distance (g) and the focal length (f) are given
    :param f: Focal length
    :param g: Object distance
    :return: b, the distance where the CCD should be placed
    """
    b = (f * g) / (g - f)
    return b

# Use the function to calculate the distance where the CCD should be placed when the focal length is 15mm
# and the object distance is 0.1, 1, 5 and 15 meters

# What is 15mm in meters?
# Focal length in meters
f = 15 / 1000

# Object distances in meters
object_distances = [0.1, 1, 5, 15, 50, 100, 1000]

for g in object_distances:
    b = camera_b_distance(f, g)
    print(f"The distance where the CCD should be placed when the object distance is {g} meters is {b:.4f} meters")


