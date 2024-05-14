### Setup ###
import numpy as np

array = np.array([
    [208, 208, 208, 202, 202, 202],
    [231, 231, 231, 231, 193, 193],
    [193, 193, 193, 167, 167, 167],
    [167, 36, 36, 36, 36, 36],
    [36, 40, 40, 217, 217, 217],
    [25, 25, 25, 25, 25, 25]
])

### Exercise 20 ###


# Make a function that compresses the array using gray-scale run-length encoding
def compress(array):
    compressed = []
    count = 1
    for i in range(1, len(array)):
        if array[i] == array[i - 1]:
            count += 1
        else:
            compressed.append((count, array[i - 1]))
            count = 1
    compressed.append((count, array[-1]))
    return compressed


# Compress the array
compressed = compress(array.flatten())

# Print the compressed array
print(compressed)


### Exercise 21 ###

# Apply a threshold of 230 to the array
threshold = 230
thresholded = array.copy()
thresholded[thresholded <= threshold] = 0
thresholded[thresholded > threshold] = 1

# Print the thresholded array
#print(thresholded)

# Encode the thresholded array using a chain code
chain_code = []
directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
direction = 2
x, y = np.where(thresholded == 1)
x, y = x[0], y[1]

while True:
    chain_code.append(direction)
    x += directions[direction][0]
    y += directions[direction][1]
    if x == 0 or x == thresholded.shape[0] - 1 or y == 0 or y == thresholded.shape[1] - 1:
        break
    if thresholded[x + directions[(direction + 1) % 4][0], y + directions[(direction + 1) % 4][1]] == 1:
        direction = (direction + 1) % 4
    elif thresholded[x + directions[direction][0], y + directions[direction][1]] == 0:
        direction = (direction - 1) % 4

# Print the chain code
#print(chain_code)


### Exercise 22 ###
# Apply a threshold of 215 to the array
threshold = 215
thresholded = array.copy()
thresholded[thresholded <= threshold] = 0
thresholded[thresholded > threshold] = 1

# Print the thresholded array
print(thresholded)

# [1; (0, 3)], [4; (3, 5)]
