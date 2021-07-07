# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 11:41:34 2021

@author: David Nguyen
"""

import imageio
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from math import cos, sin, sqrt
from sklearn.metrics import mean_squared_error


# Transform mask into array of coordinate vectors to the landmarks
def find_all_max(mask):
    (size_x, size_y) = mask.shape
    
    mask_value = np.amax(mask)
    
    mask_vectors = []
    
    for ii in range(0, size_x):
        for jj in range(0, size_y):
            if mask[ii, jj] == mask_value:
                mask_vectors.append([jj, ii])
                
    mask_vectors = np.array(mask_vectors)
    
    return mask_vectors

# RMSE comparison of reference letter R with a rotated camera version
def letter_rotation(theta, ref, cam):
    # Number of points
    nn = ref.shape[0]
    
    # Initialization
    rotated_cam = np.zeros((nn,2))
    
    # Rotation matrix
    rotation_matrix = np.array([[cos(theta), -sin(theta)],
                                [sin(theta), cos(theta)]])
    
    # Apply rotation
    for ii in range(0, nn):
        rotated_cam[ii,:] = np.dot(rotation_matrix, cam[ii,:])
    
    # Mean squared error
    MSE = mean_squared_error(ref, rotated_cam)
    
    # Root mean squared error (smoother function, better for optimization)
    RMSE = sqrt(MSE)
    
    return RMSE


# Horizontal flip 
def flip_horizontal(points, grid_size_x):
    # Number of landmarks
    number_of_points = points.shape[0]
    
    # Apply the flip in place
    for ii in range(number_of_points):
        points[ii, 0] = abs(points[ii, 0] - grid_size_x)
    
    return True


# Import landmarks for reference letter R on DMD
dmd_size_x = 1920
dmd_size_y = 1200
dmd_letter_R = np.genfromtxt('DMD_R_5Points.csv',delimiter=',',
                             skip_header=1)

# Import landmarks for reference letter R as seen on Camera 1
camera_size_x = 2304
camera_size_y = 2304
camera_letter_R = np.genfromtxt('Camera1_R_5Points.csv',delimiter=',',
                                skip_header=1)

# Flip the reference letter R on DMD
if not flip_horizontal(dmd_letter_R, dmd_size_x):
    print('Error: something went wrong when applying the horizontal flip.')
    
# Calculate center of reference R
ref_center_x = dmd_letter_R[:,0].mean()
ref_center_y = dmd_letter_R[:,1].mean()
ref_center_x = np.repeat(ref_center_x, dmd_letter_R.shape[0])
ref_center_y = np.repeat(ref_center_y, dmd_letter_R.shape[0])

# Center reference R from DMD to zero
centered_ref_R = dmd_letter_R[:,0] - ref_center_x
centered_ref_R = np.stack((centered_ref_R,
                           dmd_letter_R[:,1] - ref_center_y),
                          axis=1)

# Calculate center of R on Camera 1
center_x = camera_letter_R[:,0].mean()
center_y = camera_letter_R[:,1].mean()
center_x = np.repeat(center_x, camera_letter_R.shape[0])
center_y = np.repeat(center_y, camera_letter_R.shape[0])

# Center letter R from Camera 1 to zero
centered_camera_R = camera_letter_R[:,0] - center_x
centered_camera_R = np.stack((centered_camera_R,
                              camera_letter_R[:,1] - center_y),
                             axis=1)

# Calculate norm of reference R
ref_norm = np.linalg.norm(centered_ref_R)

# Calculate norm of R on Camera 1
norm = np.linalg.norm(centered_camera_R)

# Scale letter R from Camera 1 onto reference
scaled_camera_R = ref_norm/norm * centered_camera_R

# Find rotation angle theta
res = scipy.optimize.minimize(letter_rotation,0,
                              args=(centered_ref_R,scaled_camera_R),
                              method="L-BFGS-B")
theta = res.x[0]

# Define rotation matrix with theta
rotation_matrix = np.array([[cos(theta), -sin(theta)],
                                [sin(theta), cos(theta)]])

# Apply rotation angle to letter R from Camera 1
rotated_camera_R = (rotation_matrix@scaled_camera_R.T).T

# Translate letter R from Camera 1 to DMD reference center
scaled_camera_R[:,0] = rotated_camera_R[:,0] + ref_center_x
scaled_camera_R[:,1] = rotated_camera_R[:,1] + ref_center_y

# Plot reference scatter points from DMD
plt.figure()
plt.title('DMD Space')
plt.scatter(dmd_letter_R[:,0], dmd_letter_R[:,1],
            label='Flipped letter R on DMD', s=10)
plt.scatter(scaled_camera_R[:,0],scaled_camera_R[:,1],
            label='Camera landmarks with transform', s=50, facecolors='none',
            edgecolors='r')
plt.xlim((0, dmd_size_x))
plt.ylim((0, dmd_size_y))
plt.gca().invert_yaxis()
plt.grid()
plt.legend()
plt.show()

# Open mask from Camera
# mask_camera = imageio.imread('Mask_Camera1.tif')
mask_camera = imageio.imread('C1.tif')
mask_camera = np.array(mask_camera)

# Find coordinates of landmarks
mask_vectors = find_all_max(mask_camera)

# Remove distance to center in Camera
mask_vectors[:,0] = mask_vectors[:,0] - np.repeat(center_x[0],
                                                  mask_vectors.shape[0])
mask_vectors[:,1] = mask_vectors[:,1] - np.repeat(center_y[0],
                                                  mask_vectors.shape[0])

# Scale vectors
mask_vectors = ref_norm/norm * mask_vectors

# Rotate vectors
rotated_vectors = (rotation_matrix@mask_vectors.T).T

# Move vectors to center of DMD
rotated_vectors[:,0] = rotated_vectors[:,0] + np.repeat(ref_center_x[0],
                                                        mask_vectors.shape[0])
rotated_vectors[:,1] = rotated_vectors[:,1] + np.repeat(ref_center_y[0],
                                                        mask_vectors.shape[0])

# Apply flip
if not flip_horizontal(rotated_vectors, dmd_size_x):
    print('Error: something went wrong when applying the horizontal flip.')

# Create the DMD mask
DMD_mask = 255 * np.ones((dmd_size_y, dmd_size_x))

for ii in range(0, rotated_vectors.shape[0]):
    if rotated_vectors[ii, 0] > -1 and rotated_vectors[ii, 0] < dmd_size_x:
        if rotated_vectors[ii, 1] > -1 and rotated_vectors[ii, 1] < dmd_size_y:
            DMD_mask[int(round(rotated_vectors[ii, 1])),
                     int(round(rotated_vectors[ii, 0]))] = 0

imageio.imwrite('DMD_Mask.png', DMD_mask.astype(np.uint8))

# Display mask from Camera
plt.figure()
plt.title('Mask as seen on Camera')
plt.imshow(mask_camera)
plt.show()

# Display DMD mask
plt.figure()
plt.imshow(DMD_mask)
plt.show()

# Plot reference scatter points as seen on Camera 1
# plt.figure()
# plt.title('Reference letter R on camera (landmarks)')
# plt.scatter(camera_letter_R[:,0], camera_letter_R[:,1])
# plt.xlim((0, camera_size_x))
# plt.ylim((0, camera_size_y))
# plt.gca().invert_yaxis()
# plt.grid()
# plt.show()