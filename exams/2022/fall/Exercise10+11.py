### Exercise 10 ###
import numpy as np

# Average intensity of the cows
cows = np.array([26, 46, 33, 23, 35, 28, 21, 30, 38, 43])

# Average intensity of the sheep
sheep = np.array([67, 27, 40, 60, 39, 45, 27, 67, 43, 50, 37, 100])

# Compute the intensity threshold dividing cows and sheep using a minimum distance classifier
# The threshold is the mid-point between the two class value averages
threshold = (cows.mean() + sheep.mean()) / 2

# Fit Gaussians to the two sets of the data to do a parametric classification
mu_cows = np.mean(cows)
std_cows = np.std(cows)

mu_sheep = np.mean(sheep)
std_sheep = np.std(sheep)


def gaussian(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Print the threshold value
print(threshold)

# Print the values of the two Gaussians
print(gaussian(38, mu_cows, std_cows))
print(gaussian(38, mu_sheep, std_sheep))
