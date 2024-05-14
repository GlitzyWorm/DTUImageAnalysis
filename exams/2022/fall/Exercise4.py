### Exercise 4 ###
import numpy as np

# Initial table
org = [[64, 94, 21, 19, 31],
       [38, 88, 30, 23, 92],
       [81, 55, 47, 17, 43],
       [53, 62, 23, 23, 18],
       [35, 59, 84, 44, 90]]

# Accumulator
acc = np.zeros((5, 5))

# Copy the first row of the table to the accumulator
acc[0] = org[0]

# Go through each row of the table and update the accumulator.
# For each cell in the accumulator, add the value of the current cell
# plus the minimum value of the cell above or to the left or right of the top cell.
for i in range(1, 5):
    for j in range(5):
        acc[i, j] = org[i][j] + min(acc[i - 1][max(0, j - 1):min(5, j + 2)])

print(acc)
