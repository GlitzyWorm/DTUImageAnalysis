import numpy as np

from scipy.stats import norm

# Define the means and standard deviations of the three distributions
mean1, std1 = 3, 5
mean2, std2 = 7, 2
mean3, std3 = 15, 5

# Recalculate the intersection of two normal distributions
def intersection(mean1, std1, mean2, std2):
    # Calculate the intersection point (assuming equal height)
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = mean2/(std2**2) - mean1/(std1**2)
    c = mean1**2 /(2*std1**2) - mean2**2 / (2*std2**2) - np.log(std2/std1)
    return np.roots([a, b, c])

# Calculate intersections again
intersection1 = intersection(mean1, std1, mean2, std2)
intersection2 = intersection(mean2, std2, mean3, std3)

# Print the results
print(intersection1)
print(intersection2)