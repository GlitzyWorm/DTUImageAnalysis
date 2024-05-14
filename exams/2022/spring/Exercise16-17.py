### Setup ###
import math

import numpy as np
from matplotlib import pyplot as plt

### Exercise 16 ###

# Define the mean and standard deviation for the three classes
bad_mean = 25
bad_std = 10

medium_mean = 52
medium_std = 2

high_mean = 150
high_std = 30

# Plot the three Gaussians
x = np.linspace(0, 200, 1000)
bad_pdf = 1 / (bad_std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - bad_mean) / bad_std) ** 2)
medium_pdf = 1 / (medium_std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - medium_mean) / medium_std) ** 2)
high_pdf = 1 / (high_std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - high_mean) / high_std) ** 2)

plt.plot(x, bad_pdf, label='Bad')
plt.plot(x, medium_pdf, label='Medium')
plt.plot(x, high_pdf, label='High')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.title('Gaussian distributions of three classes')
plt.show()

# Parametric threshold calculations
s1a = medium_std
s2a = high_std
m1a = medium_mean
m2a = high_mean

# Calculate thresholds
term = -s1a**2 * s2a**2 * (2 * m2a * m1a - m2a**2 - 2 * s2a**2 * math.log10(s2a/s1a)/math.log10(math.e) - m1a**2 + 2 * s1a**2 * math.log10(s2a/s1a)/math.log10(math.e))
th2pPar = (s1a**2 * m2a - s2a**2 * m1a + math.sqrt(term)) / (-s2a**2 + s1a**2)
th2nPar = (s1a**2 * m2a - s2a**2 * m1a - math.sqrt(term)) / (-s2a**2 + s1a**2)

# Plot thresholds
plt.plot([th2pPar, th2pPar], [0, 0.1], 'c', label='Positive Threshold')
plt.plot([th2nPar, th2nPar], [0, 0.1], '--c', label='Negative Threshold')

# Display plot with legend
plt.legend()
plt.show()

# Output the threshold
print(f"Answer: Pixel value separating medium from high: {th2nPar:.1f}")


### Exercise 17 ###
# Calculate the threshold for the bad and medium classes using a minimum distance classifier
# The threshold is the mid-point between the two class value averages
threshold = (bad_mean + medium_mean) / 2

# Print the threshold value
print(f"Answer: Pixel value separating bad from medium: {threshold:.1f}")

