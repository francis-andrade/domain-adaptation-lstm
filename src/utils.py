import logging
import cv2
import sys
from scipy.ndimage import zoom
import numpy as np

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
        image = zoom(image, (60/image.shape[0], 88/image.shape[1], 1))
        image = image.reshape(3, 60, 88)
        image_array.append(image)
        success,image = vidcap.read()
        

    return image_array

def gauss2d(shape, center, gamma, out_shape=None):
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
    G = np.exp(-gamma * ((x - x0)**2 + (y - y0)**2))  # Gaussian kernel centered in (x0, y0)
    return G/np.sum(G)  # normalized so it sums to 1

def density_map(shape, centers, gammas, out_shape=None):
    if out_shape is None:
        D = np.zeros(shape)
    else:
        D = np.zeros(out_shape)
    for i, (x, y) in enumerate(centers):
        D += gauss2d(shape, (x, y), gammas[i], out_shape=out_shape)
    return D