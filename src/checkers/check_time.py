"""
Module that checks how much time it takes to load the model.
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import settings
from loaders.load_webcamt import CameraData, CameraTimeData, FrameData, VehicleData
import loaders.load_webcamt
from loaders.load_ucspeds import VideoDataUCS, FrameDataUCS
import loaders.load_ucspeds
import utils.utils
import utils.transformations
import time

if __name__ == '__main__':
    
    if settings.DATASET == 'webcamt':
        data, data_insts = loaders.load_webcamt.load_insts(settings.PREFIX_DATA, 1000)
    elif settings.DATASET == 'ucspeds':
        data, data_insts = loaders.load_ucspeds.load_insts(settings.PREFIX_DATA, 1000)

    if settings.TEMPORAL:
        data_insts = utils.utils.group_sequences(data_insts, settings.SEQUENCE_SIZE)


    no_batches = 0
    transforms = []
    
    domain_insts  = data_insts
    train_loader = utils.utils.multi_data_loader(domain_insts, 10, settings.PREFIX_DATA, settings.PREFIX_DENSITIES, data, transforms, shuffle=False)
    
    start_time = time.time()

    counts_register = []
    for i in range(len(data_insts)):
        counts_register.append([])
    i = 0
    #train_insts = list(train_loader)
    for batch_insts, batch_densities, batch_counts, batch_masks in train_loader:
        i += 1
        print(i)
        for j in range(len(batch_insts)):
            if settings.USE_MASK:
                counts = np.sum(batch_densities[j]*batch_masks[j], axis=(1,2,3))
            else:
                counts = batch_counts[j]
            counts_register[j] += counts.tolist()

    end_time = time.time()

    print("Time Taken: ", end_time - start_time)
                    
        