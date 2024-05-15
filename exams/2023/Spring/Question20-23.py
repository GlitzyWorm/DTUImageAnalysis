### Setup ###
import glob

import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from sklearn import decomposition

in_dir = "data/PizzaPCA/training/"

### Question 20-23 ###

all_images = glob.glob(in_dir + "*.png")
n_samples = len(all_images)

im_org = io.imread(all_images[0])
im_shape = im_org.shape
height = im_shape[0]
width = im_shape[1]
channels = im_shape[2]
n_features = height * width * channels

data_matrix = np.zeros((n_samples, n_features))

# Load all images and store them in the data_matrix
idx = 0
for image_file in all_images:
    img = io.imread(image_file)
    flat_img = img.flatten()
    data_matrix[idx, :] = flat_img
    idx += 1

# Compute the mean of each row
average_pizza = np.mean(data_matrix, axis=0)

# Calculate the sum of squared differences from the mean
sub_data = data_matrix - average_pizza
sub_distances = np.linalg.norm(sub_data, axis=1)

# best_match = np.argmin(sub_distances)
# best_average_pizza = data_matrix[best_match, :]
worst_match = np.argmax(sub_distances)
worst_average_pizza = data_matrix[worst_match, :]

print(f"The image with the largest sum of squared differences from the mean is: {worst_match} "
      f"with a value of {sub_distances[worst_match]}")
print(f"Pizza most away from average pizza {worst_match} : {all_images[worst_match]}")

print("Computing PCA")
pizzas_pca = decomposition.PCA(n_components=5)
pizzas_pca.fit(data_matrix)

print(f"Total variation explained by first component {pizzas_pca.explained_variance_ratio_[0] * 100}")

# Transform the data matrix using the PCA
transformed_data = pizzas_pca.transform(data_matrix)

# Find the two pizzas that are the furthest away on the first principal axes.
# One in the positive direction and one in the negative direction.
first_component = transformed_data[:, 0]
max_idx = np.argmax(first_component)
min_idx = np.argmin(first_component)

print(f"Max idx {max_idx} Min idx {min_idx}")
print(f"Pizza most away from average pizza in positive direction {max_idx} : {all_images[max_idx]}")
print(f"Pizza most away from average pizza in negative direction {min_idx} : {all_images[min_idx]}")

new_dir = "data/PizzaPCA/"

# Load super_pizza.png
super_pizza = io.imread(new_dir + "super_pizza.png")

# Project the super_pizza.png image into the PCA space
transformed_super_pizza = pizzas_pca.transform(super_pizza.flatten().reshape(1, -1))
transformed_super_pizza = transformed_super_pizza.flatten()

# Find the closest pizza in the training set to the super_pizza.png image
diff = transformed_data - transformed_super_pizza
distances = np.linalg.norm(diff, axis=1)
closest_idx = np.argmin(distances)

print(f"The closest pizza to the super_pizza.png image is: {closest_idx} with a distance of {distances[closest_idx]}")
print(f"Closest pizza to super pizza {closest_idx} : {all_images[closest_idx]}")

