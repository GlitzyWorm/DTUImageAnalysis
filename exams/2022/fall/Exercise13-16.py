### Setup ###
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import decomposition
import seaborn as sns

in_dir = "data/CarPCA/"

### Exercise 13 ###

# Load car_data.txt
data = np.loadtxt(in_dir + "car_data.txt", comments="%")

# Extract the features and labels
x = data[0:203, 0:8]

# Extract individual features
wheel_base = x[:, 0]
length = x[:, 1]
width = x[:, 2]
height = x[:, 3]
curb_weight = x[:, 4]
engine_size = x[:, 5]
horsepower = x[:, 6]
highway_mpg = x[:, 7]

# Compute the mean of each feature
mean_wheel_base = np.mean(wheel_base)
mean_length = np.mean(length)
mean_width = np.mean(width)
mean_height = np.mean(height)
mean_curb_weight = np.mean(curb_weight)
mean_engine_size = np.mean(engine_size)
mean_horsepower = np.mean(horsepower)
mean_highway_mpg = np.mean(highway_mpg)

# Subtract the mean from each feature
wheel_base -= mean_wheel_base
length -= mean_length
width -= mean_width
height -= mean_height
curb_weight -= mean_curb_weight
engine_size -= mean_engine_size
horsepower -= mean_horsepower
highway_mpg -= mean_highway_mpg

# Compute standard deviation of each feature
std_wheel_base = np.std(wheel_base)
std_length = np.std(length)
std_width = np.std(width)
std_height = np.std(height)
std_curb_weight = np.std(curb_weight)
std_engine_size = np.std(engine_size)
std_horsepower = np.std(horsepower)
std_highway_mpg = np.std(highway_mpg)

# Normalize the features
wheel_base /= std_wheel_base
length /= std_length
width /= std_width
height /= std_height
curb_weight /= std_curb_weight
engine_size /= std_engine_size
horsepower /= std_horsepower
highway_mpg /= std_highway_mpg

# Combine the features into a new matrix
x_combined = np.vstack((wheel_base, length, width, height, curb_weight, engine_size, horsepower, highway_mpg)).T

# Print the first value of wheel_base
print(wheel_base[0])
print(x_combined[0, 0])

### Exercise 14 ###

# Compute the covariance matrix using the numpy function
C_X_np = np.cov(x_combined, rowvar=False)

# Compute the eigenvalues and eigenvectors of the covariance matrix
eig_val, eig_vec = np.linalg.eig(C_X_np)

# Plot the eigenvalues
v_norm = eig_val / eig_val.sum() * 100
plt.plot(v_norm)
plt.xlabel("Principal component")
plt.ylabel("Percentage of variance")
plt.show()

# Print the total variance explained by the first two principal components
print(f"Total variance explained by the first two principal components: {v_norm[0] + v_norm[1]:.2f}%")

### Exercise 15 ###

# Project the data onto the principal components and find their coordinates in PCA space
pc_proj = eig_vec.T.dot(x_combined.T)

# Plot the data in the PCA space
plt.scatter(pc_proj[0], pc_proj[1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# Print the absolute coordinates of the first data point in the PCA space
print(f"Absolute coordinates of the first data point in the PCA space: {pc_proj[0, 0]}")

### Alternative way ###

pca = decomposition.PCA()
pca.fit(x)
values_pca = pca.explained_variance_
exp_var_ratio = pca.explained_variance_ratio_
vectors_pca = pca.components_

data_transform = pca.transform(x_combined)

# Print values_pca, exp_var_ratio, and vectors_pca
# print(f"Eigenvalues from PCA:\n{values_pca}")
# print(f"Explained variance ratio:\n{exp_var_ratio[0] + exp_var_ratio[1]:.2f}")
# print(f"Principal components:\n{vectors_pca}")
# print(f"Transformed data:\n{data_transform[0, 0]}")

### Exercise 16 ###

# Extract the first three measurements after they have been projected onto the principal components
pc_proj = pc_proj[:3]

# Make a pairplot of the first three measurements after they have been projected onto the principal components
d = pd.DataFrame(pc_proj.T, columns=['PC1', 'PC2', 'PC3'])
sns.pairplot(d)
plt.show()
