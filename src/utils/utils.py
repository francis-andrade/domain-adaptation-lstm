"""
Module that implements generic utilities functions.
"""

import logging
import cv2
import sys
import numpy as np
import settings
import loaders.load_webcamt
import loaders.load_ucspeds
import torch
#import skimage.transform as SkT

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
        '''
        if image.shape[0] != 240 or image.shape[1] != 352:
            print("Here: "+filepath)
            image = zoom(image, (240/image.shape[0], 352/image.shape[1], 1))
        '''
        image_array.append(image)
        success,image = vidcap.read()
        

    return image_array

def gauss2d(shape, center, sigmax, sigmay, out_shape=None, mask=None):
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
    if mask is not None:
        G = G*mask
    sum = np.sum(G)
    if sum == 0:
        return G
    else:
        return G/sum  # normalized so it sums to 1

def density_map(shape, centers, sigmas, out_shape=None, mask=None):
    if out_shape is None:
        D = np.zeros(shape)
    else:
        D = np.zeros(out_shape)
    for i, (x, y) in enumerate(centers):
        D += gauss2d(shape, (x, y), sigmas[i][0], sigmas[i][1], out_shape, mask)    
    return D

def obtain_frame(frame, data, prefix_frames, prefix_densities, transforms, transform_id):
    if settings.DATASET == 'webcamt':
        new_frame  = loaders.load_webcamt.load_structure(True, frame[0], frame[1], frame[2], prefix_frames, frame[3])
    elif settings.DATASET == 'ucspeds':
        new_frame = loaders.load_ucspeds.load_structure(True, frame[0], frame[1], frame[2], prefix_frames, frame[3])
    if settings.DATASET == 'webcamt':
        new_density = loaders.load_webcamt.load_structure(False, frame[0], frame[1], frame[2], prefix_densities, frame[3])
    elif settings.DATASET == 'ucspeds':
        new_density = loaders.load_ucspeds.load_structure(False, frame[0], frame[1], frame[2], prefix_densities, frame[3])
    if settings.DATASET == 'webcamt':
        new_mask = loaders.load_webcamt.load_mask(data, frame[0], frame[1], frame[3])
    elif settings.DATASET == 'ucspeds':
        new_mask = loaders.load_ucspeds.load_mask(data, frame[0], frame[1], frame[3])
    
    if settings.DATASET == 'webcamt':
        new_count = len(data[frame[0]].camera_times[frame[1]].frames[frame[2]].vehicles)
    elif settings.DATASET == 'ucspeds':
        new_count = data[frame[0]][frame[1]].frames[frame[2]].count
    
    if transform_id >= 0:
        transform_id = int(transform_id)
        new_frame = transforms[transform_id][0](new_frame)
        new_density = transforms[transform_id][1](new_density)
        new_mask = transforms[transform_id][1](new_mask)
    
    return new_frame, new_density, new_mask, new_count

def multi_data_loader(inputs, batch_size, prefix_frames, prefix_densities, data, transforms = [], shuffle=True):
    """
    Both inputs, counts and densities are list of numpy arrays, containing instances and labels from multiple sources.
    """
    input_sizes = [len(input) for input in inputs]
    min_input_size = min(input_sizes)
    num_domains = len(inputs)

    indexes = []
    
    multiplier_transform = len(transforms)+1
    for i in range(num_domains):
        indexes.append(np.arange(input_sizes[i]*multiplier_transform))
        if shuffle:
            np.random.shuffle(indexes[i])

    num_blocks = int(np.ceil(float(min_input_size*multiplier_transform) / batch_size))
    #print(num_blocks)
    for j in range(num_blocks):
        batch_inputs, batch_counts, batch_densities, batch_masks = [], [], [], []
        for i in range(num_domains):
            batch_counts.append([])
            batch_inputs.append([])
            batch_densities.append([])
            batch_masks.append([])

            for k in range(j*batch_size, min((j+1)*batch_size, min_input_size*multiplier_transform)):
                if settings.TEMPORAL:
                    batch_sequence_inputs = []
                    batch_sequence_densities = []
                    batch_sequence_masks = []
                    batch_sequence_counts = []
                    for frame in inputs[i][indexes[i][k] % input_sizes[i]]:
                        #print("FRAME: ", frame)
                        if frame[0] == '0':
                            diff = int(frame[1])
                            for l in range(diff):
                                batch_sequence_inputs.append(np.zeros((3,)+settings.get_new_shape()))
                                batch_sequence_densities.append(np.zeros((1,)+settings.get_new_shape()))
                                batch_sequence_masks.append(np.zeros((1,)+settings.get_new_shape()))
                                batch_sequence_counts.append(0)
                        else:
                            transform_id = indexes[i][k] / input_sizes[i] - 1
                            new_frame, new_density, new_mask, new_count = obtain_frame(frame, data, prefix_frames, prefix_densities, transforms, transform_id)

                            batch_sequence_inputs.append(new_frame)
                            batch_sequence_densities.append(new_density)
                            batch_sequence_masks.append(new_mask)
                            batch_sequence_counts.append(new_count)

                    batch_inputs[i].append(batch_sequence_inputs)
                    batch_densities[i].append(batch_sequence_densities)
                    batch_masks[i].append(batch_sequence_masks)
                    batch_counts[i].append(batch_sequence_counts)
                else:
                    frame = inputs[i][indexes[i][k] % input_sizes[i]]
                    transform_id = indexes[i][k] / input_sizes[i] - 1
                    new_frame, new_density, new_mask, new_count = obtain_frame(frame, data, prefix_frames, prefix_densities, transforms, transform_id)

                    batch_inputs[i].append(new_frame)
                    batch_densities[i].append(new_density)
                    batch_masks[i].append(new_mask)
                    batch_counts[i].append(new_count)
                        
            batch_inputs[i] = np.array(batch_inputs[i], dtype=np.float)
            batch_densities[i] = np.array(batch_densities[i], dtype=np.float)
            batch_masks[i] = np.array(batch_masks[i], dtype=np.float)
            batch_counts[i] = np.array(batch_counts[i], dtype=np.float)

        if settings.USE_MASK:
            batch_counts = None
            
        yield batch_inputs, batch_densities, batch_counts, batch_masks


def group_sequences(inputs, sequence_size):
    num_domains = len(inputs)

   
    seq_inputs = []

    for i in range(num_domains):
        seq_inputs.append([])
        for j in range(len(inputs[i])):
            num_blocks = int(np.ceil(len(inputs[i][j]) / sequence_size))
            for k in range(num_blocks):
                seq_inputs[i].append(inputs[i][j][k*sequence_size:min((k+1)*sequence_size, len(inputs[i][j]))])
            if len(seq_inputs[i][-1]) < sequence_size:
                diff = sequence_size - len(seq_inputs[i][-1])
                seq_inputs[i][-1].append(['0', diff])
        
        seq_inputs[i] = seq_inputs[i]
    
    return seq_inputs

def eval_mdan(mdan, test_insts, batch_size, device, prefix_frames, prefix_densities, data):
    with torch.no_grad():
        mdan.eval()
        train_loader = multi_data_loader([test_insts], batch_size, prefix_frames, prefix_densities, data)
        num_insts = 0
        mse_density_sum = 0
        mse_count_sum = 0
        mae_count_sum = 0
        for batch_insts, batch_densities, batch_counts, batch_masks in train_loader:
            target_insts = torch.from_numpy(batch_insts[0] / 255.0).float().to(device)
            target_densities = torch.from_numpy(batch_densities[0]).float().to(device)
            if settings.USE_MASK:
                target_masks = torch.from_numpy(batch_masks[0]).float().to(device)
            else:
                target_masks = None

            if settings.USE_MASK:
                if settings.TEMPORAL:
                    dim = (2,3,4)
                else:
                    dim = (1,2,3)
                target_counts = torch.sum(target_densities*target_masks, dim=dim)
            else:
                target_counts = torch.from_numpy(batch_counts[0]).float().to(device)

            preds_densities, preds_counts = mdan.inference(target_insts, target_masks)
            
            mse_density_sum += torch.sum((preds_densities - target_densities)**2).item()
            mse_count_sum += torch.sum((preds_counts - target_counts)**2).item()
            mae_count_sum += torch.sum(abs(preds_counts-target_counts)).item()
            if settings.TEMPORAL: 
                N, T, C, H, W = preds_densities.shape
                size = N*T
            else:
                N, C, H, W = preds_densities.shape
                size = N
            num_insts += size
        print("NUM INSTS: ", num_insts)
        mse_density = mse_density_sum / num_insts
        mse_count = mse_count_sum / num_insts
        mae_count = mae_count_sum / num_insts      

        return mse_density, mse_count, mae_count

def concatenate_data_insts(data_insts, i):
    domain_insts = []
    
    for j in range(len(data_insts)):
        if j != i:
            domain_insts += data_insts[j]

    return [domain_insts]

def remove_augmentations(data_insts):
    new_data_insts = []
    for i in range(len(data_insts)):
        new_data_insts.append([])

        for j in range(len(data_insts[i])):
            if settings.TEMPORAL:
                if data_insts[i][j][0][3] == 'None':
                    new_data_insts[i].append(data_insts[i][j])
            else:
                if data_insts[i][j][3] == 'None':
                    new_data_insts[i].append(data_insts[i][j])
    
    return new_data_insts

def split_test_validation(domain_insts):
    size = len(domain_insts)
    indices = np.random.permutation(size)
    
    test_idx = indices[int(settings.VALIDATION_TEST_RATIO*size):]
    test_insts = []
    for idx in test_idx:
        test_insts.append(domain_insts[idx])
    
    val_idx = indices[:int(settings.VALIDATION_TEST_RATIO*size)]
    val_insts = []
    for idx in val_idx:
        val_insts.append(domain_insts[idx])
    
    return val_insts, test_insts