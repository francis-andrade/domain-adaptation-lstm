"""
Module that implements the functions related to data augmentation.
"""

import numpy as np

def rotate(shape, coordinate_x, coordinate_y, angle):
    """
    Function that given a set of coordinates of a cell square matrix, returns the new coordinates of that cell after a rotation has been applied.  
    
    Args:
        size: size of the matrix
        coordinate_x: X coordinate of the cell
        coordinate_y: Y coordinate of the cell
        angle: Angle of the rotation. Must be in [180]
    
    Returns:
        New coordinates of the cell to which a rotation has been applied.
    Raises:
        ValueError: If angle doesn't belong to [180]
    """
    H,W  = shape
    if angle == 180:
        return W - 1 - coordinate_x, H - 1 - coordinate_y
    else:
        raise ValueError('The angle of a rotation can only be one of [180]')


def symmetric(shape, coordinate_x, coordinate_y, angle_axis):
    """
    Function that given a set of coordinates of a cell square matrix, returns the new coordinates of that cell after a symmetry has been applied.  
    
    Args:
        shape: size of the matrix
        coordinate_x: X coordinate of the cell
        coordinate_y: Y coordinate of the cell
        angle: Angle of the rotation. Must be in [0, 90]
    
    Returns:
        New coordinates of the cell to which a symmetry has been applied.
    Raises:
        ValueError: If angle doesn't belong to [0, 90]
    """
    H,W  = shape
    if angle_axis == 0:
        return coordinate_x, H - 1 - coordinate_y
    elif angle_axis == 90:
        return W - 1 - coordinate_x, coordinate_y
    else:
        raise ValueError('The angle of a symmetry can only be one of [0, 90]')


def change_brightness_contrast(matrix, brightness, contrast):
    """
    Functions that transforms the brightness and contrast of a matrix.

    Args:
        matrix: Matrix to be transformed.
        brightness: new matrix brightness.
        contrast new matrix contrast
    
    Returns:
        matrix: New matrix which the original one was transformed to.
    """
    matrix = matrix * (contrast/127+1) - contrast + brightness
    matrix = np.clip(matrix, 0, 255)
    return matrix

def transform_matrix(matrix, function, angle):
    """
    Function that applies a transformation (rotation or symmetry) to a matrix  
    
    Args:
        matrix: Matrix to be transformed
        function: Function that defines the transformation to apply (rotate or symmetry)
        angle: Angle of the transformation
    
    Returns:
        New matrix to which the original matrix was transformed to.
    Raises:
        ValueError: If matrix is empty (i.e. has size 0)
    """
    if len(matrix) == 0:
        raise ValueError('The matrix must have size bigger than 0')

    new_matrix = np.empty(matrix.shape, dtype = 'float')
    for coordinate_y in range(len(matrix)):
        for coordinate_x in range(len(matrix[coordinate_y])):
            new_x, new_y = function(matrix.shape, coordinate_x, coordinate_y, angle)
            new_matrix[new_y][new_x] = matrix[coordinate_y][coordinate_x]
    return new_matrix    



def transform_matrix_channels(matrix, function, angle):
    """
    Function that applies a transformation (rotation or symmetry) to a matrix with channels.  
    
    Args:
        matrix: Matrix to be transformed
        function: Function that defines the transformation to apply (rotate or symmetry)
        angle: Angle of the transformation
    
    Returns:
        New matrix to which the original matrix was transformed to.
    Raises:
        ValueError: If matrix is empty (i.e. has size 0)
    """
    C, H, W = matrix.shape
    new_matrix = np.empty(matrix.shape, dtype='float')
    for channel in range(C):
         new_matrix[channel] = transform_matrix(matrix[channel], function, angle)
    return new_matrix