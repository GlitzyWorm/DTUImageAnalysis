### Setup ###
import numpy as np
from scipy.linalg import inv, logm, det
import matplotlib.pyplot as plt


### Helper Function ###
def LDA(Input, Target, Priors=None):
    # Determine size of input data
    n, m = Input.shape

    # Discover and count unique class labels
    ClassLabel = np.unique(Target)
    k = len(ClassLabel)

    # Initialize
    nGroup = np.nan * np.ones(k)  # Group counts
    GroupMean = np.nan * np.ones((k, m))  # Group sample means
    PooledCov = np.zeros((m, m))  # Pooled covariance
    W = np.nan * np.ones((k, m + 1))  # model coefficients

    # Loop over classes to perform intermediate calculations
    for i in range(k):
        # Establish location and size of each class
        Group = (Target == ClassLabel[i])
        nGroup[i] = np.sum(Group)

        # Calculate group mean vectors
        GroupMean[i, :] = np.mean(Input[Group, :], axis=0)

        # Accumulate pooled covariance information
        if nGroup[i] > 1:
            PooledCov += ((nGroup[i] - 1) / (n - k)) * np.cov(Input[Group, :], rowvar=False)

    # Assign prior probabilities
    if Priors is not None:
        PriorProb = Priors
    else:
        PriorProb = nGroup / n

    # Loop over classes to calculate linear discriminant coefficients
    for i in range(k):
        # Intermediate calculation for efficiency
        # This replaces:  GroupMean(g,:) * inv(PooledCov)
        Temp = np.dot(GroupMean[i, :], inv(PooledCov))

        # Constant
        W[i, 0] = -0.5 * np.dot(Temp, GroupMean[i, :]) + np.log(PriorProb[i])

        # Linear coefficients
        W[i, 1:] = Temp

    return W


### Exercise 18 ###

# Training examples
# Passive (class 1)
X1 = np.array([1.2, 2.9, 1.7, 1.8, 3.2, 3.1])
Y1 = np.array([1.1, 0.4, -2.7, -0.3, 1.3, -0.9])

# Erupting (class 2)
X2 = np.array([0.5, 1.4, 2.7, 2])
Y2 = np.array([1.7, -2.1, -0.8, 0.5])

# Shape input
Input = np.vstack((np.hstack((X1, X2)), np.hstack((Y1, Y2)))).T
# Make class labels of class 1 and 2
Target = np.concatenate((np.zeros(len(X1)), np.ones(len(X2))))

# Plot the training examples
plt.figure()
plt.scatter(X1, Y1, c='red', marker='o', label='Class 1: Passive')
plt.scatter(X2, Y2, c='blue', marker='o', label='Class 2: Erupting')
plt.title("Training Examples")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()

# Use the LDA function to find W
W = LDA(Input, Target)

# Calculate linear log-scores for training data
L = np.dot(np.hstack((np.ones((len(Input), 1)), Input)), W.T)

# Calculate class probabilities
P = np.exp(L) / np.sum(np.exp(L), axis=1, keepdims=True)

# Plot which training points are classified correctly
# C1: Normal
q = np.where(P[:len(X1), 0] > 0.5)[0]
plt.scatter(X1[q], Y1[q], c='red', marker='x', label='Correctly Classified Class 1')

# C2: Volcanoes that are wrongly classified
q = np.where(P[len(X1):, 1] <= 0.5)[0]
plt.scatter(X2[q], Y2[q], c='blue', marker='x', label='Wrongly Classified Class 2')
plt.legend()

# Select the wrongly segmented class 1 training examples
result = P[len(X1) + q, 1]

# Print the results
print(f"Answer: Two wrongly classified with probabilities: {result[0]:.2f}, {result[1]:.2f}")

plt.show()
