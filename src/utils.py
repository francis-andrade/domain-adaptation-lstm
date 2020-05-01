import logging
import cv2
import sys
import numpy as np
import settings

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
    if out_shape is not None:
        sigmax, sigmay = sigmax / Wo, sigmay / Ho
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

    num_blocks = int(np.ceil(float(min_input_size) / batch_size))
    for j in range(num_blocks):
        batch_inputs, batch_counts, batch_densities = [], [], []
        for i in range(num_domains):
                batch_inputs.append(inputs[i][indexes[i][j*batch_size:(j+1)*batch_size]])
                batch_counts.append(counts[i][indexes[i][j*batch_size:(j+1)*batch_size]])
                batch_densities.append(densities[i][indexes[i][j*batch_size:(j+1)*batch_size]])

        yield batch_inputs, batch_densities, batch_counts

def multi_data_loader_temporal(inputs, densities, counts, batch_size, sequence_size=None):
    
    num_domains = len(inputs)
    shape = inputs[0][0][0].shape

    if sequence_size is None:
        seq_inputs, seq_counts, seq_densities = inputs, counts, densities 
        max_lens = [np.max([len(inputs[i][j] for j in range(len(inputs[i])))]) for i in range(num_domains)]
        for i in range(num_domains):
            for j in range(len(inputs[i])):
                if len(inputs[i][j]) < max_lens[i]:
                    seq_inputs[i][j].append(np.zeros(shape))
                    seq_counts[i][j].append(np.zeros(shape))
                    seq_densities[i][j].append(np.zeros(shape))
    else:
        seq_inputs, seq_counts, seq_densities = [], [], []
        for i in range(num_domains):
            seq_inputs.append([])
            seq_counts.append([])
            seq_densities.append([])
            for j in range(len(inputs[i])):
                seq_inputs[i].append(inputs[i][j*sequence_size:(j+1)*sequence_size])
                seq_counts[i].append(inputs[i][j*sequence_size:(j+1)*sequence_size])
                seq_densities[i].append(inputs[i][j*sequence_size:(j+1)*sequence_size])

                if len(seq_inputs[i][-1]) < sequence_size:
                    while(len(seq_inputs[i][-1]) < sequence_size):
                        seq_inputs[i][j].append(np.zeros(shape))
                        seq_counts[i][j].append(np.zeros(shape))
                        seq_densities[i][j].append(np.zeros(shape))


    return multi_data_loader(seq_inputs, seq_counts, seq_densities)
    