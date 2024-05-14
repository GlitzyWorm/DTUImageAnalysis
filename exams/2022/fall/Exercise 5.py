### Exercise 5 ###
import numpy as np

grey = 178 + 60 + 155 + 252
white1 = 168 + 217 + 159 + 223
white2 = 97 + 136 + 32 + 108

result = grey - white1 - white2
print(result)

### Exercise 6 ###

array = [[33, 12, 110, 122, 204, 218, 25, 231],
         [200, 53, 81, 187, 145, 135, 221, 169],
         [220, 120, 107, 6, 39, 12, 108, 201],
         [114, 168, 217, 178, 60, 97, 136, 16],
         [253, 159, 223, 155, 252, 32, 108, 86],
         [131, 68, 68, 69, 129, 244, 174, 119],
         [93, 51, 93, 122, 44, 105, 71, 112],
         [75, 149, 45, 233, 24, 64, 146, 56]]

# Create a new array with the same dimensions as the original array and fill it with zeros
haar = np.zeros((8, 8))

# Make the integral image of the original array
# In an integral image the pixel value is:
# – The sum of pixel above it and to the left of it in the original image
# – Including the pixel itself

# First row
haar[0, 0] = array[0][0]
for j in range(1, 8):
    haar[0, j] = haar[0, j - 1] + array[0][j]

# First column
for i in range(1, 8):
    haar[i, 0] = haar[i - 1, 0] + array[i][0]

# The rest of the image
for i in range(1, 8):
    for j in range(1, 8):
        haar[i, j] = array[i][j] + haar[i - 1, j] + haar[i, j - 1] - haar[i - 1, j - 1]

# Print the value at (2, 2) in the integral image
print(haar[2, 2])
