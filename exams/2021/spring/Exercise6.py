################
# DOESN'T WORK #
################






### Setup ###
import numpy as np
from skimage import io
from sklearn.decomposition import PCA

in_dir = "data/"

### Q.6 ###

# Load car1.jpg to car5.jpg from data folder
img1 = io.imread(in_dir + "car1.jpg")
img2 = io.imread(in_dir + "car2.jpg")
img3 = io.imread(in_dir + "car3.jpg")
img4 = io.imread(in_dir + "car4.jpg")
img5 = io.imread(in_dir + "car5.jpg")

# Extract
x = np.array([img1.flatten(), img2.flatten(), img3.flatten(), img4.flatten(), img5.flatten()])

# Calculate the mean of each feature
mean_feat = np.mean(x, axis=0)

# Subtract the mean from the data
data = x - mean_feat

# Calculate the covariance matrix
N = data.shape[0]
C_X = 1/(N-1) * np.matmul(data.T, data)

# Calculate the eigenvectors and eigenvalues of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(C_X)

# Plot percent variance explained
v_norm = eigenvalues / eigenvalues.sum() * 100

# Print the percent variance explained for the first principal component
print(f"Percent variance explained for the first principal component: {v_norm[0]}")
