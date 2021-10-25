# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 11:41:34 2021

@author: David Nguyen
"""
# Modules
import imageio
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from math import cos, sin, sqrt
from scipy.sparse.linalg.matfuncs import _smart_matrix_product
from sklearn.metrics import mean_squared_error

# Detect reference mask on camera (feature under development)
def detect_reference(camera_image):
    return True

# Make a point into a circle at specified coordinates in a DMD mask
# The default size is for the grid, and the alternate size is for
# the three landmarks
def expand_point(mask, x_coord, y_coord, landmark=False):
    if landmark:
        mask[y_coord-2:y_coord+3, x_coord-2:x_coord+3] = 0
        mask[y_coord-1:y_coord+2, x_coord-3:x_coord+4] = 0
        mask[y_coord-3:y_coord+4, x_coord-1:x_coord+2] = 0
        mask[y_coord-4:y_coord+5, x_coord] = 0 
        mask[y_coord, x_coord-4:x_coord+5] = 0                 
    else:
        mask[y_coord-1:y_coord+2, x_coord-1:x_coord+2] = 0
        mask[y_coord, x_coord-2:x_coord+3] = 0
        mask[y_coord-2:y_coord+3, x_coord] = 0
        
    # Return a success flag
    return True

# Create a reference mask of size 1920 x 1200 (no error trapping) for
# calibrating the DMD (feature under development)
def create_ref_mask(size_x, size_y):
    # Initialize mask template (reverse coordinates to comply with Python)
    reference_mask = 255 * np.ones((size_y, size_x))
    
    # Number of points along one dimension (specify an odd number)
    n_points = 15
    
    # Spacing between points
    spacing = 25
    
    # Grid center
    center_x = size_x/2 - 1
    center_y = size_y/2 - 1
    
    # Iterate over the points
    for xx in range(n_points):
        for yy in range(n_points):
            x_coord = int(spacing * (xx - (n_points - 1) / 2) + center_x)
            y_coord = int(spacing * (yy - (n_points - 1) / 2) + center_y)
            
            # Expand the points to a larger circle in 3 locations
            expand_flag = x_coord == center_x and y_coord == center_y
            expand_flag = expand_flag or (xx == 0 and yy == 0)
            expand_flag = expand_flag or (xx == 0 and yy == (n_points-1) / 2)
            
            if expand_flag:
                expand_point(reference_mask, x_coord, y_coord, landmark=True)
            else:
                expand_point(reference_mask, x_coord, y_coord)
    
    # Write the DMD reference mask to a PNG file    
    imageio.imwrite('DMD_Reference_Mask.png', reference_mask.astype(np.uint8))
    
    # Return a success flag
    return True

# Takes an array <mask> (image) as a mask in camera space (2304 x 2304) and
# transform each maximum (sort of binarization) pixel into a 2D vector of X, Y
# coordinates (in camera space)
def find_all_max(mask):
    # Find the size of the mask (limited to 2304 x 2304 currently)
    (size_x, size_y) = mask.shape
    
    # Determine the maximum value in the mask
    mask_value = np.amax(mask)
    
    # Initialize the output 2D vector list 
    vector_count = 0
    mask_vectors = np.zeros((np.count_nonzero(mask == mask_value),2))
    
    # Find, and store all vectors corresponding to a maximum in the mask
    for ii in range(0, size_x):
        for jj in range(0, size_y):
            if mask[ii, jj] == mask_value:
                # Notice the reverse coordinates here for consistancy with
                # how Python treats the arrays
                mask_vectors[vector_count,0] = jj
                mask_vectors[vector_count,1] = ii
                vector_count += 1
    
    # Return a 2D vector array of the mask in camera space
    return mask_vectors

# Takes a reference <ref> of (nn) vectors and compare it to another set <cam>
# rotated by an angle <theta>. Returns the RMSE between the two sets of
# vectors. This is used to optimize the angle <theta> which minize the RMSE
# between the two models
def letter_rotation(theta, ref, cam):
    # Number of vectors composing the reference model
    nn = ref.shape[0]
    
    # Initialize an array for the rotated <cam> model
    rotated_cam = np.zeros((nn,2))
    
    # Rotation matrix
    rotation_matrix = np.array([[cos(theta), -sin(theta)],
                                [sin(theta), cos(theta)]])
    
    # Apply rotation (there might be a better algebraic way of computing
    # the dot-product for each vector...)
    for ii in range(0, nn):
        rotated_cam[ii,:] = np.dot(rotation_matrix, cam[ii,:])
    
    # Calculate the mean squared error from sklearn.metrics
    MSE = mean_squared_error(ref, rotated_cam)
    
    # Calculate the square root of the  MSE
    # (smoother function, better for optimization)
    RMSE = sqrt(MSE)
    
    # Return the RMSE
    return RMSE

# Horizontal flip of a 2D vector list <vectors> based on a <grid_size_x>
# THIS FLIP IS APPLIED IN PLACE => <vectors> is directly modified!
def flip_horizontal(vectors, grid_size_x):
    # Number of vectors
    number_of_vectors = vectors.shape[0]
    
    # Apply the flip in place!
    for ii in range(number_of_vectors):
        vectors[ii, 0] = abs(vectors[ii, 0] - grid_size_x)
    
    # Return true if successful (no error trapping)
    return True

def find_affine_parameters(
    dmd_size_x=1920,
    dmd_size_y=1200,
    dmd_R_file='DMD_R_5Points.csv',
    camera_R_file='Camera1_R_5Points.csv',
    ):
    dmd_letter_R = np.genfromtxt(
        'DMD_R_5Points.csv',
        delimiter=',',
        skip_header=1
        )
    camera_letter_R = np.genfromtxt(
        'Camera1_R_5Points.csv',
        delimiter=',',
        skip_header=1
        )
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

    return ref_norm,\
        rotation_matrix,\
        ref_center_x,\
        ref_center_y,\
        center_x,\
        center_y,\
        norm

    
if __name__ == '__main__':

    dmd_size_x=1920
    dmd_size_y=1200
    # retrieval of the affine transform
    ref_norm, rotation_matrix, ref_center_x, ref_center_y,\
        center_x, center_y, norm = find_affine_parameters()
    shift_steps = 3
    max_shift = 30
    # need the top-left corner of the ROI with respect to full camera chip
    row_0, col_0 = 100, 100
    images_path = '/home/ngc/Desktop/ROI'
    output_path = '/home/ngc/Desktop/ROI'
    starting_image = np.array(imageio.imread(images_path + '/roi.png'))
    offsets = np.zeros((1, 2))

    if starting_image.shape != (2304, 2304):
        offsets[0, 0] = row_0
        offsets[0, 1] = col_0
        # Minus one as Python arrays start at zero
        offsets -= 1
    iter_name = 0 #used in the for cycle to name to new images
    for i in range(0, max_shift, shift_steps):

        shifted_image = np.zeros(starting_image.shape)
        # move ROI to the right
        shifted_image[:, i:] = starting_image[:, :starting_image.shape[1] - i]

        # Find coordinates of landmarks
        mask_vectors = find_all_max(shifted_image)

        # Apply ROI if necessary (by default, offsets is a zero-vector)
        mask_vectors = np.add(mask_vectors, offsets)

        # Remove distance to center in Camera
        mask_vectors[:,0] = mask_vectors[:,0] - np.repeat(center_x[0],
                                                        mask_vectors.shape[0])
        mask_vectors[:,1] = mask_vectors[:,1] - np.repeat(center_y[0],
                                                    mask_vectors.shape[0])

        # Scale vectors
        mask_vectors = ref_norm/norm * mask_vectors

        # Rotate vectors
        rotated_vectors = (rotation_matrix @ mask_vectors.T).T

        # Move vectors to center of DMD
        rotated_vectors[:, 0] = rotated_vectors[:,0] + np.repeat(ref_center_x[0],
                                                        mask_vectors.shape[0])
        rotated_vectors[:, 1] = rotated_vectors[:,1] + np.repeat(ref_center_y[0],
                                                        mask_vectors.shape[0])
        # Apply flip
        if not flip_horizontal(rotated_vectors, dmd_size_x):
            print('Error: something went wrong applying the horizontal flip.')

        # Create the DMD mask
        DMD_mask = 255 * np.ones((dmd_size_y, dmd_size_x))

        for ii in range(0, rotated_vectors.shape[0]):
            if (rotated_vectors[ii, 0] > -1) and \
                (rotated_vectors[ii, 0] < dmd_size_x):
                if (rotated_vectors[ii, 1] > -1 and rotated_vectors[ii, 1] < dmd_size_y):
                    DMD_mask[int(round(rotated_vectors[ii, 1])),
                            int(round(rotated_vectors[ii, 0]))] = 0

        imageio.imwrite(
            output_path + '/DMD_Mask_' + 
            str(iter_name) + '.png', DMD_mask.astype(np.uint8))
        iter_name += 1
