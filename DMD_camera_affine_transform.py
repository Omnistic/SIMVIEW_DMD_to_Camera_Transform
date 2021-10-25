from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import imageio

def letter_R(rows, cols, scale=50):
    """Create a list of coordinates.
    The points are distrubuted as a capital R.
        - rows = rows max dimension
        - cols = columns max dimension
        - scale = distance between the points (in pixels)
    """
    center_row = int(rows / 2)
    center_col = int(cols / 2)
    letter = []

    letter.append((center_row - 2 * scale, center_col - scale))
    letter.append((center_row - 2 * scale, center_col))
    letter.append((center_row - 2 * scale, center_col + scale))

    letter.append((center_row - scale, center_col - scale))
    letter.append((center_row - scale, center_col))
    letter.append((center_row - scale, center_col + scale))

    letter.append((center_row, center_col - scale))
    letter.append((center_row, center_col + scale))

    letter.append((center_row + scale, center_col - scale))
    letter.append((center_row + scale, center_col + 2 * scale))

    letter.append((center_row + 2 * scale, center_col - scale))
    letter.append((center_row + 2 * scale, center_col + 3 * scale))
    
    return letter

def image_to_DMD(image):
    """From a camera image to the DMD plane.
    It takes care of inversion and rotation.
    There two transormation are always happening, and depende
    on the geometry of the light-sheet.
        - image = camera image as 2D array
    """
    
    return sp.ndimage.rotate(image[:, ::-1], 45)

if __name__ == '__main__':

    folder = '/home/ngc/Desktop/cali_tests'
    side_points = 5 # will make a 10x10 square grid of points
    step = 100 # pixels, distance between the points
    point_size = 5 # a point_size x point_size square ROI
    row_0, col_0 = 1000, 1000 # top left corner of the grid in camera coords

    image_size = 2304

    total_target = np.zeros((image_size, image_size), dtype=np.uint8)

    for i in range(side_points):
        for j in range(side_points):
            point = np.zeros((image_size, image_size), dtype=np.uint8)
            point[
                row_0 + i * step - point_size : row_0 + i * step + point_size,
                col_0 + j * step - point_size:col_0 + j * step + point_size
            ] = 255
            total_target += point
            imageio.imwrite(folder + '/pnt_' + str(i) + '_' + str(j) + '.png',
            point)
    imageio.imwrite(folder + '/total_grid.png', total_target)


            



