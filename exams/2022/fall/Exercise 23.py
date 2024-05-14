### Exercise 23 ###

# Size of one frame
image_size = 1024 * 768 * 3

# 30 megabytes in bytes
max_size = 30 * 1024 * 1024
# The way the answer did it
# max_size = 30 * 1000 * 1000

# How many frames per second can be analyzed?
fps = max_size / image_size

# Print the result
print(fps)

# The algorithm uses 54 milliseconds to analyze one frame.
# How many frames can be analyzed in one second?
frames_per_second = 1000 / 54
# The way the answer did it
# frames_per_second = 1 / 0.054

# Print the result
print(frames_per_second)