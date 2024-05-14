### Setup ###
import numpy as np
from matplotlib import pyplot as plt
from sklearn import decomposition

in_dir = "data/PCAData/"

### Exercise 14 ###

# Load soccer_data.txt
data = np.loadtxt(in_dir + "soccer_data.txt", comments="%")

# Extract the features and labels
x = data[0:2964, 0:6]

pca = decomposition.PCA()
pca.fit(x)
values_pca = pca.explained_variance_
exp_var = pca.explained_variance_ratio_
vectors_pca = pca.components_

data_transform = pca.transform(x)

# Print the maximum absolute value of all the projected data
max_abs = np.max(np.abs(data_transform))
print(f"Maximum absolute value of all the projected data: {max_abs}")

### Exercise 15 ###

# Plot the explained variance ratio
plt.plot(exp_var)
plt.xlabel("Principal component")
plt.ylabel("Explained variance ratio")
plt.show()

