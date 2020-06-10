import logging
import cv2
import sys
import numpy as np
import settings
import load_webcamt
import load_ucspeds
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

def multi_data_loader(inputs, counts, batch_size, prefix_frames, prefix_densities, data, transforms = [], shuffle=True):
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
    
    new_counts = []
    for i in range(num_domains):
        original_count = np.copy(counts[i])
        new_counts.append(original_count)
        for t in range(len(transforms)):
            new_counts[i] = np.concatenate((new_counts[i], original_count))

    num_blocks = int(np.ceil(float(min_input_size*multiplier_transform) / batch_size))
    for j in range(num_blocks):
        batch_inputs, batch_counts, batch_densities, batch_masks = [], [], [], []
        for i in range(num_domains):
                    batch_counts.append(new_counts[i][indexes[i][j*batch_size:(j+1)*batch_size]])
                    if settings.TEMPORAL:
                        sequence_size = settings.SEQUENCE_SIZE
                        batch_inputs.append(np.zeros((0,sequence_size,3)+settings.get_new_shape()))
                        batch_densities.append(np.zeros((0,sequence_size,1)+settings.get_new_shape()))
                        batch_masks.append(np.zeros((0,sequence_size,1)+settings.get_new_shape()))
                    else:
                        batch_inputs.append(np.zeros((0,3)+settings.get_new_shape()))
                        batch_densities.append(np.zeros((0,1)+settings.get_new_shape()))
                        batch_masks.append(np.zeros((0,1)+settings.get_new_shape()))

                    for k in range(j*batch_size, min((j+1)*batch_size, min_input_size*multiplier_transform)):
                        if settings.TEMPORAL:
                            batch_sequence_inputs = []
                            batch_sequence_densities = []
                            batch_sequence_masks = []
                            for frame in inputs[i][indexes[i][k] % input_sizes[i]]:
                                if frame[0] == '0':
                                    diff = int(frame[1])
                                    for l in range(diff):
                                        batch_sequence_inputs.append(np.zeros((3,)+settings.get_new_shape()))
                                        batch_sequence_densities.append(np.zeros((1,)+settings.get_new_shape()))
                                        batch_sequence_masks.append(np.zeros((1,)+settings.get_new_shape()))
                                else:
                                    if settings.DATASET == 'webcamt':
                                        new_frame  = load_webcamt.load_structure(True, frame[0], frame[1], frame[2], prefix_frames, frame[3])
                                    elif settings.DATASET == 'ucspeds':
                                        new_frame = load_ucspeds.load_structure(True, frame[0], frame[1], frame[2], prefix_frames, frame[3])
                                    if settings.DATASET == 'webcamt':
                                        new_density = load_webcamt.load_structure(False, frame[0], frame[1], frame[2], prefix_densities, frame[3])
                                    elif settings.DATASET == 'ucspeds':
                                        new_density = load_ucspeds.load_structure(False, frame[0], frame[1], frame[2], prefix_densities, frame[3])
                                    if settings.DATASET == 'webcamt':
                                        new_mask = np.array([load_webcamt.load_mask(data, frame[0], frame[1])])
                                    elif settings.DATASET == 'ucspeds':
                                        new_mask = np.array([load_ucspeds.load_mask(data, frame[0], frame[1])])
                                    transform_id = indexes[i][k] / input_sizes[i] - 1
                                    if transform_id >= 0:
                                        transform_id = int(transform_id)
                                        new_frame = transforms[transform_id][0](new_frame)
                                        new_density = transforms[transform_id][1](new_density)
                                        new_mask = transforms[transform_id][1](new_mask)

                                    batch_sequence_inputs.append(new_frame)
                                    batch_sequence_densities.append(new_density)
                                    batch_sequence_masks.append(new_mask)
                            batch_inputs[i] = np.concatenate((batch_inputs[i], np.array([batch_sequence_inputs])))
                            batch_densities[i] = np.concatenate((batch_densities[i], np.array([batch_sequence_densities])))
                            batch_masks[i] = np.concatenate((batch_masks[i], np.array([batch_sequence_masks])))
                        else:
                            frame = inputs[i][indexes[i][k] % input_sizes[i]]
                            if settings.DATASET == 'webcamt':
                                new_frame = load_webcamt.load_structure(True, frame[0], frame[1], frame[2], prefix_frames,  frame[3])
                            elif settings.DATASET == 'ucspeds':
                                new_frame = load_ucspeds.load_structure(True, frame[0], frame[1], frame[2], prefix_frames, frame[3])
                            if settings.DATASET == 'webcamt':
                                new_density = load_webcamt.load_structure(False, frame[0], frame[1], frame[2], prefix_densities, frame[3])
                            elif settings.DATASET == 'ucspeds':
                                new_density = load_ucspeds.load_structure(False, frame[0], frame[1], frame[2], prefix_densities, frame[3])
                            if settings.DATASET == 'webcamt':
                                new_mask = np.array([load_webcamt.load_mask(data, frame[0], frame[1])])
                            elif settings.DATASET == 'ucspeds':
                                new_mask = np.array([load_ucspeds.load_mask(data, frame[0], frame[1])])

                            transform_id = indexes[i][k] / input_sizes[i] - 1
                            if transform_id >= 0:
                                transform_id = int(transform_id)
                                new_frame = transforms[transform_id][0](new_frame)
                                new_density = transforms[transform_id][1](new_density)
                                new_mask = transforms[transform_id][1](new_mask)

                            new_frame = np.array([new_frame])
                            new_density = np.array([new_density])
                            new_mask = np.array([new_mask])
                            batch_inputs[i] = np.concatenate((batch_inputs[i], new_frame))
                            batch_densities[i] = np.concatenate((batch_densities[i], new_density))
                            batch_masks[i] = np.concatenate((batch_masks[i], new_mask))

        if settings.USE_MASK:
            batch_counts = None
            
        yield batch_inputs, batch_densities, batch_counts, batch_masks



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
                    seq_inputs[i][-1] = np.concatenate((seq_inputs[i][-1], np.array([['0', diff, '0', '0']])))
                    seq_counts[i][-1] = np.concatenate((seq_counts[i][-1] , np.zeros((diff,))))
        
            seq_counts[i] = np.array(seq_counts[i])
            seq_inputs[i] = np.array(seq_inputs[i])
    
    return seq_inputs, seq_counts

def eval_mdan(mdan, test_insts, test_counts, batch_size, device, prefix_frames, prefix_densities, data):
    with torch.no_grad():
        mdan.eval()
        train_loader = multi_data_loader([test_insts], [test_counts], batch_size, prefix_frames, prefix_densities, data)
        num_insts = 0
        mse_density_sum = 0
        mse_count_sum = 0
        mae_count_sum = 0
        for batch_insts, batch_densities, batch_counts, batch_masks in train_loader:
            target_insts = torch.from_numpy(np.array(batch_insts[0], dtype=np.float)).float().to(device)
            target_densities = torch.from_numpy(np.array(batch_densities[0], dtype=np.float)).float().to(device)
            if settings.USE_MASK:
                target_masks = torch.from_numpy(np.array(batch_masks[0],  dtype=np.float)).float().to(device)
            else:
                target_masks = None

            if settings.USE_MASK:
                if settings.TEMPORAL:
                    dim = (2,3,4)
                else:
                    dim = (1,2,3)
                target_counts = torch.sum(target_densities*target_masks, dim=dim)
            else:
                target_counts = torch.from_numpy(np.array(batch_counts[0], dtype=np.float)).float().to(device)

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

def concatenate_data_insts(data_insts, data_counts, i):
    domain_insts, domain_counts = np.empty((0,)+data_insts[0][0].shape), np.empty((0,)+data_counts[0][0].shape)
    
    for j in range(len(data_insts)):
        if j != i:
            domain_insts = np.concatenate((domain_insts, data_insts[j]))
            domain_counts = np.concatenate((domain_counts, data_counts[j]))

    return [domain_insts], [domain_counts]

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