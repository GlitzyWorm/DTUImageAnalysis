import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import decomposition
from sklearn.decomposition import PCA

### Setup

in_dir = "data/"

txt_name = "irisdata.txt"

iris_data = np.loadtxt(in_dir + txt_name, comments="%")
x = iris_data[0:150, 0:4]

n_feat = x.shape[1]
n_obs = x.shape[0]

# print(f"Number of features: {n_feat} and number of observations: {n_obs}")

### Q.1 ###

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
# plt.plot(v_norm)
# plt.xlabel("Principal component")
# plt.ylabel("Percent variance explained")
# plt.ylim([0, 100])
# plt.show()

# Print the percent variance explained for the first two principal components
# print(f"Percent variance explained for the first two principal components: {v_norm[0] + v_norm[1]}")

### Q.2 ###

# Compute the signals
signals = eigenvectors.T @ data.T
# print(signals[:, 0])

