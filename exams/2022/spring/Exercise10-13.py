### Setup ###
import numpy as np
from skimage import io

in_dir = "data/ImagePCA/"

### Functions ###
def sortem(V, D):
    """
    Assumes the columns of V are vectors to be sorted along with the
    diagonal elements of D.
    """

    if V is None or D is None:
        raise ValueError('Must specify vector matrix and diag value matrix')

    # Extract the diagonal elements of D
    dvec = np.diag(D)

    # Sort the diagonal elements in descending order
    sorted_indices = np.argsort(dvec)[::-1]

    # Create new sorted matrices
    NV = np.zeros_like(V)
    ND = np.zeros_like(D)

    for i in range(D.shape[0]):
        ND[i, i] = D[sorted_indices[i], sorted_indices[i]]
        NV[:, i] = V[:, sorted_indices[i]]

    return NV, ND


def pc_evectors(A, numvecs):
    """
    Get the top numvecs eigenvectors of the covariance matrix of A,
    using Turk and Pentland's trick for numrows >> numcols.
    Returns the eigenvectors as the columns of Vectors and a vector of
    ALL the eigenvalues in Values.
    """

    if A is None or numvecs is None:
        raise ValueError('usage: pc_evectors(A, numvecs)')

    nexamp = A.shape[1]

    # Compute the "average" vector
    print('Computing average vector and vector differences from avg...')
    Psi = np.mean(A, axis=1)

    # Compute difference with average for each vector
    A = A - Psi[:, np.newaxis]

    # Get the patternwise (nexamp x nexamp) covariance matrix
    print('Calculating L=A\'A')
    L = A.T @ A

    # Get the eigenvectors (columns of Vectors) and eigenvalues (diag of Values)
    print('Calculating eigenvectors of L...')
    values, vectors = np.linalg.eigh(L)

    # Sort the vectors/values according to size of eigenvalue
    print('Sorting evectors/values...')
    vectors, values = sortem(vectors, values)

    # Convert the eigenvectors of A'*A into eigenvectors of A*A'
    print('Computing eigenvectors of the real covariance matrix...')
    vectors = A @ vectors

    # Normalize Vectors to unit length, kill vectors corr. to tiny evalues
    num_good = 0
    for i in range(nexamp):
        vectors[:, i] = vectors[:, i] / np.linalg.norm(vectors[:, i])
        if values[i] < 0.00001:
            values[i] = 0
            vectors[:, i] = np.zeros(vectors.shape[0])
        else:
            num_good += 1

    if numvecs > num_good:
        print(f'Warning: numvecs is {numvecs}; only {num_good} exist.')
        numvecs = num_good

    vectors = vectors[:, :numvecs]

    # Get the eigenvalues out of the diagonal matrix and normalize them
    values = values / (nexamp - 1)

    return vectors, values, Psi


# Example usage
A = np.random.rand(10, 5)  # Example data matrix
numvecs = 3  # Number of eigenvectors to retrieve
vectors, values, Psi = pc_evectors(A, numvecs)
print("Eigenvectors:\n", vectors)
print("Eigenvalues:\n", values)
print("Psi:\n", Psi)

### Exercise 11 ###

# Load the images spoon1-6.png
spoon1 = io.imread(in_dir + 'spoon1.png', as_gray=True)
spoon2 = io.imread(in_dir + 'spoon2.png', as_gray=True)
spoon3 = io.imread(in_dir + 'spoon3.png', as_gray=True)
spoon4 = io.imread(in_dir + 'spoon4.png', as_gray=True)
spoon5 = io.imread(in_dir + 'spoon5.png', as_gray=True)
spoon6 = io.imread(in_dir + 'spoon6.png', as_gray=True)

# Apply a threshold of 100 to each image
threshold = 100
spoon1 = spoon1 > threshold
spoon2 = spoon2 > threshold
spoon3 = spoon3 > threshold
spoon4 = spoon4 > threshold
spoon5 = spoon5 > threshold
spoon6 = spoon6 > threshold

# Perform PCA on the images
A = np.array([spoon1.ravel(), spoon2.ravel(), spoon3.ravel(), spoon4.ravel(), spoon5.ravel(), spoon6.ravel()])
numvecs = 2
vectors, values, Psi = pc_evectors(A, numvecs)

# How many percent of the variance is explained by the first eigenvector?
print(f"Percent explained variance: {values[0] / np.sum(values) * 100:.2f}%")