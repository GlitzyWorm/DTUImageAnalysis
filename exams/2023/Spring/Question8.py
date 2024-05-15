### Setup ###
import numpy as np
import SimpleITK as sitk


### Helper Function ###
def rotation_matrix(pitch, roll, yaw, deg=False):
    """
    Return the rotation matrix associated with the Euler angles roll, pitch, yaw.

    Parameters
    ----------
    pitch : float
        The rotation angle around the x-axis.
    roll : float
        The rotation angle around the y-axis.
    yaw : float
        The rotation angle around the z-axis.
    deg : bool, optional
        If True, the angles are given in degrees. If False, the angles are given
        in radians. Default: False.
    """
    if deg:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

    R_x = np.array([[1, 0, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch), 0],
                    [0, np.sin(pitch), np.cos(pitch), 0],
                    [0, 0, 0, 1]])

    R_y = np.array([[np.cos(roll), 0, np.sin(roll), 0],
                    [0, 1, 0, 0],
                    [-np.sin(roll), 0, np.cos(roll), 0],
                    [0, 0, 0, 1]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                    [np.sin(yaw), np.cos(yaw), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    R = np.dot(np.dot(R_x, R_y), R_z)

    return R


### Question 8 ###

# Create an affine transformation with the following parameters:
# - Roll: 30 degrees
# - Translation: 10 in x
# - Yaw: 10 degrees
# All the matrices should be 4x4

# Create the rotation matrix
angle = 30
roll_rad = np.deg2rad(angle)
rot_matrix = rotation_matrix(0, roll_rad, 0)

# Create the translation matrix
trans_matrix = np.array([[1, 0, 0, 10],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

# Create the yaw matrix
angle = 10
yaw_rad = np.deg2rad(angle)
yaw_matrix = rotation_matrix(0, 0, yaw_rad)

# Combine the matrices
affine_matrix = yaw_matrix @ (trans_matrix @ rot_matrix)
print(affine_matrix)
