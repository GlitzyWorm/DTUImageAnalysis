### Setup ###
import numpy as np
from sklearn import decomposition

in_dir = "data/GlassPCA/"

### Question 4 ###

# Load the txt file glass_data.txt
glass_data = np.loadtxt(in_dir + "glass_data.txt", comments="%")

# Extract the data into a matrix X
x = glass_data[0:214, 0:9]

# Compute the mean of the x
mn = np.mean(x, axis=0)
data = x - mn

# Extract each feature
ri = data[:, 0]
na = data[:, 1]
mg = data[:, 2]
al = data[:, 3]
si = data[:, 4]
k  = data[:, 5]
ca = data[:, 6]
ba = data[:, 7]
fe = data[:, 8]

# Compute the minimum and maximum values of the features and find the difference
min_ri = np.min(ri)
max_ri = np.max(ri)
diff_ri = max_ri - min_ri

min_na = np.min(na)
max_na = np.max(na)
diff_na = max_na - min_na

min_mg = np.min(mg)
max_mg = np.max(mg)
diff_mg = max_mg - min_mg

min_al = np.min(al)
max_al = np.max(al)
diff_al = max_al - min_al

min_si = np.min(si)
max_si = np.max(si)
diff_si = max_si - min_si

min_k = np.min(k)
max_k = np.max(k)
diff_k = max_k - min_k

min_ca = np.min(ca)
max_ca = np.max(ca)
diff_ca = max_ca - min_ca

min_ba = np.min(ba)
max_ba = np.max(ba)
diff_ba = max_ba - min_ba

min_fe = np.min(fe)
max_fe = np.max(fe)
diff_fe = max_fe - min_fe

# Divide each feature by the difference between the maximum and minimum values
ri_norm = ri / diff_ri
na_norm = na / diff_na
mg_norm = mg / diff_mg
al_norm = al / diff_al
si_norm = si / diff_si
k_norm = k / diff_k
ca_norm = ca / diff_ca
ba_norm = ba / diff_ba
fe_norm = fe / diff_fe

# Combine the normalized features into a matrix X_norm
X_norm = np.column_stack((ri_norm, na_norm, mg_norm, al_norm, si_norm, k_norm, ca_norm, ba_norm, fe_norm))

# Compute the PCA
pca = decomposition.PCA()
pca.fit(X_norm)
values_pca = pca.explained_variance_
exp_var_ratio = pca.explained_variance_ratio_
vectors_pca = pca.components_

# Print the total variance explained by the first three principal components
print(f"Explained variance ratio: {(exp_var_ratio[0] + exp_var_ratio[1] + exp_var_ratio[2]) * 100:.2f}")


### Question 5 ###

# Compute the covariance matrix using the numpy function
C_X_np = np.cov(X_norm, rowvar=False)

print(f"Covariance matrix computed by numpy: {C_X_np[0][0]}")


### Question 6 ###

# Print the first value of na_norm
print(f"First value of si_norm: {na_norm[0]}")

### Question 7 ###

data_transform = pca.transform(X_norm)

# Compute the absolute value of all the projected values and the maximum value is printed
max_abs = np.max(np.abs(data_transform))
print(f"Maximum absolute value of all the projected data: {max_abs}")
