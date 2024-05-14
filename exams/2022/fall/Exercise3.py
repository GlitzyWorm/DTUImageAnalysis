# x cos theta + y sin theta = rho -> y = (rho - x cos theta) / sin theta
import numpy as np


def hough(x):
    return (0.29 - x * np.cos(np.deg2rad(151))) / np.sin(np.deg2rad(151))


print(hough(7))
print(hough(9))
print(hough(6))
print(hough(3))
