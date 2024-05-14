import math

import numpy as np

### Exercise 6 ###

# Define the points
points = np.array([[1, 13], [2, 9], [4, 4], [5, 12], [5, 5], [5, 2], [7, 7], [10, 4], [12, 9], [13, 9]])

# Define the thetas and rhos
thetas = [45, 90, -45, -45, 0]
rhos = [9.9, 9.0, -4.9, 0.0, 5.0]

for theta, rho in zip(thetas, rhos):
    print(f"Theta = {theta}, rho = {rho}")
    count = 0
    for point in points:
        x = point[0]
        y_real = point[1]
        y = -x * (math.cos(math.radians(theta)) / math.sin(math.radians(theta))) + \
            rho * 1 / math.sin(math.radians(theta))
        if abs(y - y_real) < 1:
            count += 1
            print(f"Found pair (x,y) = ({x}, {y:.0f}). Real y = {y_real}.")
        # print(f"Found pairs (x,y) = ({x}, {y:.0f}). Real y = {y_real}.")
    print(f"Count: {count}\n\n")


### Exercise 7 ###

# In Hough space, the line equation is given by:
# ρ = x cos(θ) + y sin(θ)

# Given 𝜃 = 0∘:
# cos(0) = 1
# sin(0) = 0

# So the equation becomes:
# ρ=x
# 10=x

# This is a vertical line x=10, and it passes through all points where x=10.
# Which only one point does.
