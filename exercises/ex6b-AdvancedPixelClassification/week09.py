import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from data import *

# Imported from data/LDA
import numpy as np


def LDA(X, y):
    """
    Linear Discriminant Analysis.

    A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule.
    Assumes equal priors among classes

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target values.

    Returns
    -------
    W : array-like of shape (n_classes, n_features+1)
        Weights for making the projection. First column is the constants.

    Last modified: 11/11/22, mcbo@dtu.dk
    """

    # Determine size of input data
    n, m = X.shape
    # Discover and count unique class labels
    class_label = np.unique(y)
    k = len(class_label)

    # Initialize
    n_group = np.zeros((k, 1))  # Group counts
    group_mean = np.zeros((k, m))  # Group sample means
    pooled_cov = np.zeros((m, m))  # Pooled covariance
    W = np.zeros((k, m + 1))  # Model coefficients

    for i in range(k):
        # Establish location and size of each class
        group = np.squeeze(y == class_label[i])
        n_group[i] = np.sum(group.astype(np.double))

        # Calculate group mean vectors
        group_mean[i, :] = np.mean(X[group, :], axis=0)

        # Accumulate pooled covariance information
        pooled_cov = pooled_cov + ((n_group[i] - 1) / (n - k)) * np.cov(X[group, :], rowvar=False)

    # Assign prior probabilities
    prior_prob = n_group / n

    # Loop over classes to calculate linear discriminant coefficients
    for i in range(k):
        # Intermediate calculation for efficiency
        temp = group_mean[i, :][np.newaxis] @ np.linalg.inv(pooled_cov)

        # Constant
        W[i, 0] = -0.5 * temp @ group_mean[i, :].T + np.log(prior_prob[i])

        # Linear
        W[i, 1:] = temp

    return W



in_dir = 'data/'
in_file = 'ex6_ImagData2Load.mat'
data = sio.loadmat(in_dir + in_file)
ImgT1 = data['ImgT1']
ImgT2 = data['ImgT2']
ROI_GM = data['ROI_GM'].astype(bool)
ROI_WM = data['ROI_WM'].astype(bool)

# Exercise 1: Plot the T1 and T2 images, their 1D and 2D histograms and scatter plots
fig, axs = plt.subplots(3, 2)
axs[0, 0].imshow(ImgT1, cmap='gray')
axs[0, 0].set_title('T1 image')
axs[0, 1].imshow(ImgT2, cmap='gray')
axs[0, 1].set_title('T2 image')

# 1D histograms
axs[1, 0].hist(ImgT1.ravel(), bins=25, density=True, alpha=1, label='T1')
axs[1, 1].hist(ImgT2.ravel(), bins=25, density=True, alpha=1, label='T2')

# 2D histograms
axs[2, 0].hist2d(ImgT1.ravel(), ImgT2.ravel(), bins=10, cmap='viridis')
axs[2, 0].set_title('2D histogram')
axs[2, 0].set_xlabel('T1')
axs[2, 0].set_ylabel('T2')

# Scatter plot
axs[2, 1].scatter(ImgT1.ravel(), ImgT2.ravel(), alpha=0.5)
axs[2, 1].set_title('Scatter plot')
axs[2, 1].set_xlabel('T1')
axs[2, 1].set_ylabel('T2')

plt.subplots_adjust(hspace=0.5)
plt.show()

# Exercise 2: Place the ROI_GM and ROI_WM into variables C1 and C2. Show C1 and C2 as images.
C1 = ROI_GM
C2 = ROI_WM

fig, axs = plt.subplots(1, 2)
axs[0].imshow(C1, cmap='gray')
axs[0].set_title('C1')
axs[1].imshow(C2, cmap='gray')
axs[1].set_title('C2')
plt.show()

# Exercise 3: For each binary training ROI find the corresponding training examples in ImgT1 and ImgT2
# Tips: If you are a MATLAB-like programming lover, you may use the np.argwhere() function appropriately to return
# the index to voxels in the image fulfilling e.g. intensity values >0 hence belong to a given class.
# Name the index variables qC1 and qC2, respectively.
qC1 = np.argwhere(C1)
qC2 = np.argwhere(C2)

# What is the difference between the 1D histogram of the training examples and the 1D histogram of the whole image?
# Is the difference expected?
# Plot the 1D histograms of the training examples and the whole image.

# 1D histograms
fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(ImgT1.ravel(), bins=25, density=True, alpha=1, label='T1')
axs[0, 1].hist(ImgT2.ravel(), bins=25, density=True, alpha=1, label='T2')

axs[1, 0].hist(ImgT1[qC1[:, 0], qC1[:, 1]], bins=25, density=True, alpha=1, label='T1')
axs[1, 1].hist(ImgT2[qC1[:, 0], qC1[:, 1]], bins=25, density=True, alpha=1, label='T2')

plt.show()

# Exercise 4: Make a training data vector (X) and target class vector (T) as input for the LDA() function.
# T and X should have the same length of data points.
#
# X: Training data vector should first include all data points for class 1 and then the data points for class 2.
# Data points are the two input features ImgT1, ImgT2
#
# T: Target class identifier for X where '0' are Class 1 and a '1' is Class 2.
#
# Tip: Read the documentation of the provided LDA function to understand the expected input dimensions.

# X
X = np.vstack((ImgT1.ravel(), ImgT2.ravel())).T

# T
T = np.zeros(X.shape[0])

# Mark the class 2 data points with '1'
T[qC2[:, 0] * ImgT2.shape[1] + qC2[:, 1]] = 1

# Exercise 5: Make a scatter plot of the training points of the two input features for class 1 and
# class 2 as green and black circles, respectively. Add relevant title and labels to axis
plt.scatter(X[T == 0, 0], X[T == 0, 1], c='g', label='Class 1')
plt.scatter(X[T == 1, 0], X[T == 1, 1], c='k', label='Class 2')
plt.title('Training data')
plt.xlabel('T1')
plt.ylabel('T2')
plt.legend()
plt.show()

# Exercise 6: Train the linear discriminant classifier using the Fisher discriminant function and
# estimate the weight-vector coefficient W (i.e. w0 and w) for classification given X and T by using the W=LDA() function.
# The LDA function outputs W=[[w01, w1]; [w02, w2]] for class 1 and 2 respectively.

# LDA function
W = LDA(X, T)

# Exercise 7: Apply the linear discriminant classifier
# i.e. perform multi-modal classification using the trained weight-vector W
#  for each class: It calculates the linear score Y for all image data points within the brain slice
#  i.e. y(x) = w + w0. Actually, y(x) is the log(P(Ci|x)).

Xall = np.c_[ImgT1.ravel(), ImgT2.ravel()]
Y = np.c_[np.ones((len(Xall), 1)), Xall] @ W.T

# Exercise 8: Perform multi-modal classification: Calculate the posterior probability
# i.e. P(C1|x) of a data point belonging to class 1
#
# Note: Using Bayes [Eq 1]: Since y(x) is the log of the posterior probability [Eq2] we take exp(y(x))
#  to get P(C1X) = P(X|μ, σ)P(C1) and divide with the marginal probability P(X) as normalisation factor.
PosteriorProb = np.clip(np.exp(Y) / np.sum(np.exp(Y),1)[:, np.newaxis], 0, 1)

# Exercise 9: Apply segmentation: Find all voxels in the T1w and T2w image with P(C1|X) > 0.5 as belonging to Class 1.
# You may use the np.where() function. Similarly, find all voxels belonging to class 2.

# Class 1
C1_seg = np.zeros(ImgT1.shape)
C1_seg.ravel()[np.where(PosteriorProb[:, 0] > 0.5)] = 1

# Class 2
C2_seg = np.zeros(ImgT1.shape)
C2_seg.ravel()[np.where(PosteriorProb[:, 1] > 0.5)] = 1

# Exercise 10: Show scatter plots of segmentation results as in Exercise 5.
plt.scatter(Xall[np.where(C1_seg.ravel()), 0], Xall[np.where(C1_seg.ravel()), 1], c='r', label='Class 1 seg')
plt.scatter(Xall[np.where(C2_seg.ravel()), 0], Xall[np.where(C2_seg.ravel()), 1], c='b', label='Class 2 seg')
plt.title('Segmentation results')
plt.xlabel('T1')
plt.ylabel('T2')
plt.legend()
plt.show()



