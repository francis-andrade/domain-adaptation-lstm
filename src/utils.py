import logging
import cv2
import sys
import numpy as np
import settings
import load_data
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
    sum = np.sum(G)
    if sum == 0:
        return G
    else:
        return G/sum  # normalized so it sums to 1

def density_map(shape, centers, sigmas, out_shape=None):
    if out_shape is None:
        D = np.zeros(shape)
    else:
        D = np.zeros(out_shape)
    for i, (x, y) in enumerate(centers):
        D += gauss2d(shape, (x, y), sigmas[i][0], sigmas[i][1], out_shape)
    #print(np.sum(D), len(centers))    
    return D

def multi_data_loader(inputs, counts, batch_size, prefix_frames, prefix_densities):
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
                                    new_frame = load_data.load_structure(True, frame[0], frame[1], frame[2], prefix_frames, frame[3])
                                    new_density = load_data.load_structure(False, frame[0], frame[1], frame[2], prefix_densities, frame[3])
                                    batch_sequence_inputs.append(new_frame)
                                    batch_sequence_densities.append(new_density)
                            batch_inputs[i] = np.concatenate((batch_inputs[i], np.array([batch_sequence_inputs])))
                            batch_densities[i] = np.concatenate((batch_densities[i], np.array([batch_sequence_densities])))
                        else:
                            frame = inputs[i][indexes[i][k]]
                            new_frame = np.array([load_data.load_structure(True, frame[0], frame[1], frame[2], prefix_frames, frame[3])])
                            new_density = np.array([load_data.load_structure(False, frame[0], frame[1], frame[2], prefix_densities, frame[3])])
                            batch_inputs[i] = np.concatenate((batch_inputs[i], new_frame))
                            batch_densities[i] = np.concatenate((batch_densities[i], new_density))

        yield batch_inputs, batch_densities, batch_counts



def group_sequences(inputs, counts, sequence_size = None):
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
            seq_inputs[i] = np.array(seq_inputs[i])
    
    return seq_inputs, seq_counts

num_insts = None
train_loader = None
def eval_mdan(mdan, test_insts, test_counts, batch_size, device, prefix_frames, prefix_densities):
    global num_insts, train_loader
    with torch.no_grad():
        mdan.eval()
        train_loader = multi_data_loader([test_insts], [test_counts], batch_size, prefix_frames, prefix_densities)
        num_insts = 0
        mse_density_sum = 0
        mse_count_sum = 0
        mae_count_sum = 0
        for batch_insts, batch_densities, batch_counts in train_loader:
            target_insts = torch.from_numpy(np.array(batch_insts[0], dtype=np.float)).float().to(device)
            densities = np.array(batch_densities[0], dtype=np.float)
            if settings.TEMPORAL:
                N, T, C, H, W = densities.shape 
                densities = np.reshape(densities, (N*T, C, H, W))
            target_densities = torch.from_numpy(np.array(densities, dtype=np.float)).float().to(device)
            target_counts = torch.from_numpy(np.array(batch_counts[0], dtype=np.float)).float().to(device)
            preds_densities, preds_counts = mdan.inference(target_insts)
            mse_density_sum += torch.sum((preds_densities - target_densities)**2).item()
            mse_count_sum += torch.sum((preds_counts - target_counts)**2).item()
            mae_count_sum += torch.sum(abs(preds_counts-target_counts)).item()
            num_insts += len(target_densities)
        mse_density = mse_density_sum / num_insts
        mse_count = mse_count_sum / num_insts
        mae_count = mae_count_sum / num_insts      

        return mse_density, mse_count, mae_count

'''
def show_images(plt, var_name, X, density, count, shape=None):
    labels = ['img {} count = {} | '.format(i, int(cnti)) for i, cnti in enumerate(count)]

    if shape is not None:
        N = X.shape[0]  # N, C, H, W
        X, density = X.transpose(2, 3, 0, 1), density.transpose(2, 3, 0, 1)  # H, W, N, C (format expected by skimage)
        X, density = SkT.resize(X, (shape[0], shape[1], N, 3)), SkT.resize(density, (shape[0], shape[1], N, 1))
        X, density = X.transpose(2, 3, 0, 1), density.transpose(2, 3, 0, 1)  # N, C, H, W
    Xh = np.tile(np.mean(X, axis=1, keepdims=True), (1, 3, 1, 1))
    density = np.squeeze(density)
    density[density < 0] = 0.
    scale = np.max(density, axis=(1, 2))[:, np.newaxis, np.newaxis] + 1e-9
    density /= scale
    Xh[:, 1, :, :] *= 1 - density
    Xh[:, 2, :, :] *= 1 - density
    density = np.tile(density[:, np.newaxis, :, :], (1, 3, 1, 1))
    plt.plot(var_name + ' highlighted', Xh, labels)
    plt.plot(var_name + ' density maps', density, labels)
'''