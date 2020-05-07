import logging
import cv2
import sys
import numpy as np
import settings
from load_data import load_structure

def isInteger(str):
    try:
        int(str)
        return True
    except:
        return False

def retrieveTime(str):
    year = int(str[0:4])
    month = int(str[4:6])
    day = int(str[6:8])
    hour = int(str[9:11])

    if len(str) > 11:
        minute = int(str[12:])
    else:
        minute=-1
    
    return year, month, day, hour, minute

def retrieveTimeXML(str):
    [date, time]=str.split(" ")
    [year,month,day] = [int(s) for s in date.split("/")]
    [hour,minute,second] = [int(s) for s in time.split(":")]
    return year, month, day, hour, minute, second

def get_logger(filename):
    # Logging configuration: set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s',
                                      datefmt='%m-%d %H:%M')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # File logger
    file_handler = logging.FileHandler("{}.log".format(filename))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    # Stderr logger
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.DEBUG)
    logger.addHandler(std_handler)
    return logger

def readFramesFromVideo(filepath):
    vidcap = cv2.VideoCapture(filepath)

    image_array = []
    success,image = vidcap.read()

    while success:    
        #print(image.shape)
        '''
        if image.shape[0] != 240 or image.shape[1] != 352:
            print("Here: "+filepath)
            image = zoom(image, (240/image.shape[0], 352/image.shape[1], 1))
        '''
        image_array.append(image)
        success,image = vidcap.read()
        

    return image_array

def gauss2d(shape, center, sigmax, sigmay, out_shape=None):
    H, W = shape
    if out_shape is None:
        Ho = H
        Wo = W
    else:
        Ho, Wo = out_shape
    x, y = np.array(range(Wo)), np.array(range(Ho))
    x, y = np.meshgrid(x, y)
    x, y = x.astype(float)/Wo, y.astype(float)/Ho
    x0, y0 = float(center[0])/W, float(center[1])/H
    
    sigmax, sigmay = sigmax / W, sigmay / H
    G = np.exp(-(1/2)*(((x - x0)/sigmax)**2 + ((y - y0)/sigmay)**2))  # Gaussian kernel centered in (x0, y0)
    #return G
    return G/np.sum(G)  # normalized so it sums to 1

def density_map(shape, centers, sigmas, out_shape=None):
    if out_shape is None:
        D = np.zeros(shape)
    else:
        D = np.zeros(out_shape)
    for i, (x, y) in enumerate(centers):
        D += gauss2d(shape, (x, y), sigmas[i][0], sigmas[i][1], out_shape)
    #print(np.sum(D), len(centers))    
    return D

def multi_data_loader(inputs, densities, counts, batch_size):
    """
    Both inputs, counts and densities are list of numpy arrays, containing instances and labels from multiple sources.
    """
    input_sizes = [len(data) for data in inputs]
    min_input_size = min(input_sizes)
    num_domains = len(inputs)

    indexes = []
    for i in range(num_domains):
        indexes.append(np.arange(len(inputs[i])))
        np.random.shuffle(indexes[i])

    num_blocks = int(np.floor(float(min_input_size) / batch_size))
    for j in range(num_blocks):
        batch_inputs, batch_counts, batch_densities = [], [], []
        for i in range(num_domains):
                if settings.LOAD_MULTIPLE_FILES:
                    batch_counts.append(counts[i][indexes[i][j*batch_size:(j+1)*batch_size]])
                    if settings.TEMPORAL:
                        sequence_size = len(inputs[i][0])
                        batch_inputs.append(np.zeros((0,sequence_size,3)+settings.IMAGE_NEW_SHAPE))
                        batch_densities.append(np.zeros((0,sequence_size,1)+settings.IMAGE_NEW_SHAPE))
                    else:
                        batch_inputs.append(np.zeros((0,3)+settings.IMAGE_NEW_SHAPE))
                        batch_densities.append(np.zeros((0,1)+settings.IMAGE_NEW_SHAPE))
                    for k in range(j*batch_size, min((j+1)*batch_size, len(indexes[i]))):
                        if settings.TEMPORAL:
                            batch_sequence_inputs = []
                            batch_sequence_densities = []
                            for frame in inputs[i][indexes[i][k]]:
                                if frame[0] == '0':
                                    diff = frame[1]
                                    for l in range(diff):
                                        batch_sequence_inputs.append(np.zeros((3,)+settings.IMAGE_NEW_SHAPE))
                                        batch_sequence_densities.append(np.zeros((1,)+settings.IMAGE_NEW_SHAPE))
                                else:
                                    new_frame = load_structure(True, frame[0], frame[1], frame[2], 'first')
                                    new_density = load_structure(False, frame[0], frame[1], frame[2], 'first')
                                    batch_sequence_inputs.append(new_frame)
                                    batch_sequence_densities.append(new_density)
                            batch_inputs[i] = np.concatenate((batch_inputs[i], np.array([batch_sequence_inputs])))
                            batch_densities[i] = np.concatenate((batch_densities[i], np.array([batch_sequence_densities])))
                        else:
                            frame = inputs[i][indexes[i][k]]
                            new_frame = np.array([load_structure(True, frame[0], frame[1], frame[2], 'first')])
                            new_density = np.array([load_structure(False, frame[0], frame[1], frame[2], 'first')])
                            batch_inputs[i] = np.concatenate((batch_inputs[i], new_frame))
                            batch_densities[i] = np.concatenate((batch_densities[i], new_density))
                    
                else:
                    batch_inputs.append(inputs[i][indexes[i][j*batch_size:min((j+1)*batch_size, min_input_size)]])
                    batch_counts.append(counts[i][indexes[i][j*batch_size:min((j+1)*batch_size, min_input_size)]])
                    batch_densities.append(densities[i][indexes[i][j*batch_size:min((j+1)*batch_size, min_input_size)]])


        yield batch_inputs, batch_densities, batch_counts


def group_sequences(inputs, densities, counts, sequence_size=None):

    num_domains = len(inputs)
    input_shape = inputs[0][0][0].shape
    density_shape = densities[0][0][0].shape

    if sequence_size is None:
        seq_inputs, seq_counts, seq_densities = inputs, counts, densities 
        max_lens = [np.max([len(inputs[i][j] for j in range(len(inputs[i])))]) for i in range(num_domains)]
        for i in range(num_domains):
            for j in range(len(inputs[i])):
                if len(seq_inputs[i][-1]) < max_lens[i]:
                    diff = max_lens[i] - len(seq_inputs[i][-1])
                    seq_inputs[i][-1] = np.concatenate((seq_inputs[i][-1] , np.zeros((diff,)+input_shape)))
                    seq_counts[i][-1] = np.concatenate((seq_counts[i][-1] , np.zeros((diff,))))
                    seq_densities[i][-1] = np.concatenate((seq_densities[i][-1] , np.zeros((diff,)+density_shape)))
    else:
        seq_inputs, seq_counts, seq_densities = [], [], []
        for i in range(num_domains):
            seq_inputs.append([])
            seq_counts.append([])
            seq_densities.append([])
            for j in range(len(inputs[i])):
                num_blocks = int(np.ceil(len(inputs[i][j]) / sequence_size))
                for k in range(num_blocks):
                    seq_inputs[i].append(inputs[i][j][k*sequence_size:(k+1)*sequence_size])
                    seq_counts[i].append(counts[i][j][k*sequence_size:(k+1)*sequence_size])
                    seq_densities[i].append(densities[i][j][k*sequence_size:(k+1)*sequence_size])

                if len(seq_inputs[i][-1]) < sequence_size:
                    diff = sequence_size - len(seq_inputs[i][-1])
                    seq_inputs[i][-1] = np.concatenate((seq_inputs[i][-1] , np.zeros((diff,)+input_shape)))
                    seq_counts[i][-1] = np.concatenate((seq_counts[i][-1] , np.zeros((diff,))))
                    seq_densities[i][-1] = np.concatenate((seq_densities[i][-1] , np.zeros((diff,)+density_shape)))
            
            seq_inputs[i] = np.array(seq_inputs[i])
            seq_counts[i] = np.array(seq_counts[i])
            seq_densities[i] = np.array(seq_densities[i])


    return seq_inputs, seq_densities, seq_counts


def group_sequences_load_multiple_files(inputs, counts, sequence_size = None):
    num_domains = len(inputs)

    if sequence_size is None:
        seq_inputs, seq_counts = inputs, counts
        max_lens = [np.max([len(inputs[i][j] for j in range(len(inputs[i])))]) for i in range(num_domains)]
        for i in range(num_domains):
            for j in range(len(inputs[i])):
                if len(seq_inputs[i][-1]) < max_lens[i]:
                    diff = max_lens[i] - len(seq_inputs[i][-1])
                    seq_inputs[i][-1].append(['0', diff])
                    seq_counts[i][-1] = np.concatenate((seq_counts[i][-1] , np.zeros((diff,))))
    else:
        seq_inputs = []
        seq_counts = []
        for i in range(num_domains):
            seq_inputs.append([])
            seq_counts.append([])
            for j in range(len(inputs[i])):
                num_blocks = int(np.ceil(len(inputs[i][j]) / sequence_size))
                for k in range(num_blocks):
                    seq_inputs[i].append(inputs[i][j][k*sequence_size:min((k+1)*sequence_size, len(inputs[i][j]))])
                    seq_counts[i].append(counts[i][j][k*sequence_size:min((k+1)*sequence_size, len(inputs[i][j]))])
                if len(seq_inputs[i][-1]) < sequence_size:
                    diff = sequence_size - len(seq_inputs[i][-1])
                    seq_inputs[i][-1].append(['0', diff])
                    seq_counts[i][-1] = np.concatenate((seq_counts[i][-1] , np.zeros((diff,))))
        
            seq_counts[i] = np.array(seq_counts[i])
    
    return seq_inputs, seq_counts

def rotate(size, coordinate_x, coordinate_y, angle):
    """
    Function that given a set of coordinates of a cell square matrix, returns the new coordinates of that cell after a rotation has been applied.  
    
    Args:
        size: size of the matrix
        coordinate_x: X coordinate of the cell
        coordinate_y: Y coordinate of the cell
        angle: Angle of the rotation. Must be in [90, 180, 270]
    
    Returns:
        New coordinates of the cell to which a rotation has been applied.
    Raises:
        ValueError: If angle doesn't belong to [90, 180, 270]
    """
    if angle == 90:
        return coordinate_y, size - 1 - coordinate_x
    elif angle == 180:
        return size - 1 - coordinate_x, size - 1 - coordinate_y
    elif angle == 270:
        return size - 1 - coordinate_y, coordinate_x
    else:
        raise ValueError('The angle of a rotation can only be one of [90, 180, 270]')


def symmetric(size, coordinate_x, coordinate_y, angle_axis):
    """
    Function that given a set of coordinates of a cell square matrix, returns the new coordinates of that cell after a symmetry has been applied.  
    
    Args:
        size: size of the matrix
        coordinate_x: X coordinate of the cell
        coordinate_y: Y coordinate of the cell
        angle: Angle of the rotation. Must be in [0, 45, 90, 135]
    
    Returns:
        New coordinates of the cell to which a symmetry has been applied.
    Raises:
        ValueError: If angle doesn't belong to [0, 45, 90, 135]
    """
    if angle_axis == 0:
        return coordinate_x, size - 1 - coordinate_y
    elif angle_axis == 45:
        return coordinate_y, coordinate_x
    elif angle_axis == 90:
        return size - 1 - coordinate_x, coordinate_y
    elif angle_axis == 135:
        return size - 1 - coordinate_y, size - 1 - coordinate_x
    else:
        raise ValueError('The angle of a symmetry can only be one of [0, 45, 90, 135]')


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
        ValueError: If matrix is not square
    """
    if len(matrix) == 0:
        raise ValueError('The matrix must have size bigger than 0')
    elif not(len(matrix) == len(matrix[0])):
        raise ValueError('The matrix must be square')

    size = len(matrix)
    new_matrix = np.empty((size, size), dtype = 'float')
    for coordinate_y in range(len(matrix)):
        for coordinate_x in range(len(matrix[coordinate_y])):
            new_x, new_y = function(size, coordinate_x, coordinate_y, angle)
            new_matrix[new_y][new_x] = matrix[coordinate_y][coordinate_x]
    return new_matrix    
