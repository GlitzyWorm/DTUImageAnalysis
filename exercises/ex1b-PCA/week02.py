import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import decomposition

### Setup

# Directory containing data and images
in_dir = "exercises/ex1b-PCA/data/"

# Name of the text file containing the iris data
txt_name = "irisdata.txt"

### Exercise 1
iris_data = np.loadtxt(in_dir + txt_name, comments="%")
# x is a matrix with 50 rows and 4 columns
x = iris_data[0:50, 0:4]

# Check data dimensions
n_feat = x.shape[1]
n_obs = x.shape[0]
print(f"Number of features: {n_feat} and number of observations: {n_obs}")

### Exercise 2

# Vectors of individual features
sep_l = x[:, 0]
sep_w = x[:, 1]
pet_l = x[:, 2]
pet_w = x[:, 3]

# Compute variance of each feature
# Use ddof = 1 to make an unbiased estimate
""" var_sep_l = sep_l.var(ddof=1)
var_sep_w = sep_w.var(ddof=1)
var_pet_l = pet_l.var(ddof=1)
var_pet_w = pet_w.var(ddof=1) """

""" print(f"Variance of sepal length: {var_sep_l}")
print(f"Variance of sepal width: {var_sep_w}")
print(f"Variance of petal length: {var_pet_l}")
print(f"Variance of petal width: {var_pet_w}") """

### Exercise 3
# Compute the covariance matrix of sepal length and sepal width
""" cov_sep = 1/(n_obs-1) * np.multiply(sep_l, sep_w).sum() """

# Print 
""" print(f"Covariance matrix for sepal length and width: {cov_sep}") """

# Compute the covariance matrix of sepal length and petal length
""" cov_sep_pet = 1/(n_obs-1) * np.multiply(sep_l, pet_l).sum() """

# Print
""" print(f"Covariance matrix for sepal and petal length: {cov_sep_pet}") """

### Exercise 4
""" plt.figure() # Added this to make sure that the figure appear
# Transform the data into a Pandas dataframe
d = pd.DataFrame(x, columns=['Sepal length', 'Sepal width',
							 'Petal length', 'Petal width'])
sns.pairplot(d)
plt.show() """

### Exercise 5
mn = np.mean(x, axis=0)
data = x - mn

# Compute the covariance matrix
N = data.shape[0]  # Number of observations
C_X = 1/(N-1) * np.matmul(data.T, data)

# Compute the covariance matrix using the numpy function
C_X_np = np.cov(x, rowvar=False)

# Print the covariance matrices
""" print(f"Covariance matrix computed by hand:\n{C_X}")
print(f"Covariance matrix computed by numpy:\n{C_X_np}") """

### Exercise 6
# Compute the eigenvalues and eigenvectors of the covariance matrix
values, vectors = np.linalg.eig(C_X) # Here c_x is your covariance matrix.

### Exercise 7
""" v_norm = values  / values.sum() * 100
plt.plot(v_norm)
plt.xlabel('Principal component')
plt.ylabel('Percent explained variance')
plt.ylim([0, 100])

plt.show() """

### Exercise 8

pc_proj = vectors.T.dot(data.T)

#plt.figure() # Added this to make sure that the figure appear
# Transform the data into a Pandas dataframe
d = pd.DataFrame(pc_proj.T, columns=['PC1', 'PC2',
							 'PC3', 'PC4'])
sns.pairplot(d)
plt.show()

### Exercise 9
pca = decomposition.PCA()
pca.fit(x)
values_pca = pca.explained_variance_
exp_var_ratio = pca.explained_variance_ratio_
vectors_pca = pca.components_

data_transform = pca.transform(data)

# Print values_pca, exp_var_ratio, and vectors_pca
print(f"Eigenvalues from PCA:\n{values_pca}")
print(f"Explained variance ratio:\n{exp_var_ratio}")
print(f"Eigenvectors from PCA:\n{vectors_pca}")
print(f"Transformed data:\n{data_transform}")