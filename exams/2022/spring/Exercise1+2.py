### Setup ###
import numpy as np

### Exercise 1 ###

# Create a 5x5 array
data = np.array([
    [177, 195, 181, 30, 192],
    [81, 203, 192, 127, 65],
    [242, 48, 70, 245, 129],
    [9, 125, 173, 87, 178],
    [112, 114, 167, 149, 227]
])

# Make an empty array of the same shape
result = np.zeros(data.shape)

# Copy the first row of data to the first row of result
result[0] = data[0]

# Go through each cell from the second row to the last row.
# For each cell, take the value from the original data array
# and add the minimum value of the cell above and to the left and right of the top cell in the result array.
for i in range(1, data.shape[0]):
    for j in range(data.shape[1]):
        result[i, j] = data[i, j] + min(result[i - 1, max(j - 1, 0)], result[i - 1, j], result[i - 1, min(j + 1, data.shape[1] - 1)])

# Find the most optimal path from the top to the bottom
path = [np.argmin(result[-1])]
for i in range(data.shape[0] - 1, 0, -1):
    path.append(np.argmin(result[i - 1, max(path[-1] - 1, 0):min(path[-1] + 2, data.shape[1])]) + max(path[-1] - 1, 0))
path.reverse()

### Answer to Exercise 2 ###

# Print the path and the result
print(path)
print(result)

### Answer to Exercise 1 ###

# Make a list of the values in the result array that are on the path
path_values = [result[i, path[i]] for i in range(data.shape[0])]

# Print the values
print(np.median(path_values))
