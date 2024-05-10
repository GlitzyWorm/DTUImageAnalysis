import numpy as np


def lda(input_data, target, priors=None):
    # Determine the size of input data
    n, m = input_data.shape

    # Discover and count unique class labels
    class_labels = np.unique(target)
    k = len(class_labels)

    # Initialize
    n_group = np.zeros(k)  # Group counts
    group_means = np.zeros((k, m))  # Group sample means
    pooled_cov = np.zeros((m, m))  # Pooled covariance matrix
    w = np.zeros((k, m + 1))  # Model coefficients

    # Calculate group means and pooled covariance
    for i, label in enumerate(class_labels):
        group = (target == label)
        n_group[i] = np.sum(group)
        group_means[i, :] = np.mean(input_data[group, :], axis=0)
        # Accumulate pooled covariance
        if n_group[i] > 1:
            pooled_cov += ((n_group[i] - 1) / (n - k)) * np.cov(input_data[group, :], rowvar=False)

    # Determine prior probabilities
    if priors is not None:
        prior_prob = priors
    else:
        prior_prob = n_group / n

    # Compute linear discriminant coefficients
    for i in range(k):
        temp = np.linalg.solve(pooled_cov, group_means[i, :])
        w[i, 0] = -0.5 * np.dot(np.dot(group_means[i, :], np.linalg.inv(pooled_cov)), group_means[i, :]) + np.log(
            prior_prob[i])
        w[i, 1:] = temp

    return w


# Define the input data and target
X = np.array([[1, 1], [2.2, -3], [3.5, -1.4], [3.7, -2.7], [5, 0],
              [0.1, 0.7], [0.22, -2.1], [0.35, -0.98], [0.37, -1.89], [0.5, 0]])
T = np.array([0]*5 + [1]*5)

# Call the lda function
W = lda(X, T)

# Define ex1
ex1 = np.array([1, 1, 1])

# Calculate Y
Y = np.dot(W, ex1)

# Calculate the softmax
softmax = np.exp(Y) / np.sum(np.exp(Y))

# Print the softmax
print(softmax)
