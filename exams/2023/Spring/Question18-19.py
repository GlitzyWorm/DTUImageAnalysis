### Setup ###



### Question 18-19 ###

# RGB image size of 1600x800 pixels with 8 bits per channel
# image_size = 1600 * 800 * 8
image_size = 1600 * 800 * 3  # Why is this 3 and not 8?

# Transfer time from camera to computer (images/second)
transfer_time = 6.25

# Algorithm processing time (milliseconds/second)
processing_time = 0.230

# Number of frames per second that can be processed
fps = 1 / processing_time

# Print the minimum of transfer time and processing time
print(min(transfer_time, fps))

# How much data can be transferred in one second?
data_transfer = transfer_time * image_size

# Print the data transfer
print(data_transfer)
