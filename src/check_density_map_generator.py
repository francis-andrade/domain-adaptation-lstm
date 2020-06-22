import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import settings
from load_webcamt import CameraData, CameraTimeData, FrameData, VehicleData
import load_webcamt
from load_ucspeds import VideoDataUCS, FrameDataUCS
import load_ucspeds
import utils
import transformations


def show_plot(frame, i, X, density, count=None, mask=None):
    if frame is None:
        cid = 'dont know'
    else:
        cid = str(frame[0])+'/' + str(frame[1])+'/'+str(frame[2])
    X = np.moveaxis(X, 0, 2)
    density = np.moveaxis(density, 0, 2)

    gs = gridspec.GridSpec(3, 2)
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(X/255.0)
    ax1.set_title('Image')
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(density.squeeze(), cmap='gray')
    ax2.set_title('Density map')
    ax3 = fig.add_subplot(gs[1, 1])
    Xh = np.tile(np.mean(X, axis=2, keepdims=True), (1, 1, 3))
    Xh[:, :, 1] *= (1-density.squeeze()/np.max(density))
    Xh[:, :, 2] *= (1-density.squeeze()/np.max(density))
    ax3.imshow(Xh.astype('uint8'))
    ax3.set_title('Highlighted vehicles')

    if settings.USE_MASK:
        mask = np.moveaxis(mask, 0, 2)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow((X*mask).astype('uint8'))
        ax4.set_title('Masked Image')

        ax5 = fig.add_subplot(gs[2, :])
        Xm = np.tile(np.mean(X*mask, axis=2, keepdims=True), (1, 1, 3))
        dm = density*mask
        Xm[:, :, 1] *= (1-(dm).squeeze()/np.max(dm))
        Xm[:, :, 2] *= (1-dm.squeeze()/np.max(dm))
        ax5.imshow((Xm*mask).astype('uint8'))
        ax5.set_title('Masked Highlighted')

    mask_sum = -1
    if settings.USE_MASK:
        mask_sum = np.sum(density*mask)
    print('Image {}: cid={}, aug = {}, count={}, density_sum={:.3f}, density_mask_sum={:.3f}'.format(
        i, cid, 'dont know', count, np.sum(density), mask_sum))
    plt.show()


if __name__ == '__main__':
    #data = load_data.load_data(1, settings.WEBCAMT_NEW_SHAPE)
    #data =  load_data.load_data_from_file('first', 'proportional')

    if settings.DATASET == 'webcamt':
        data, data_insts = load_webcamt.load_insts(settings.PREFIX_DATA, 20)
    elif settings.DATASET == 'ucspeds':
        data, data_insts = load_ucspeds.load_insts(settings.PREFIX_DATA, 20)

    if settings.TEMPORAL:
        data_insts = utils.group_sequences(data_insts, settings.SEQUENCE_SIZE)

    '''
    for i in range(len(data_insts)):
        data_insts[i] = data_insts[i][::-1]
        data_counts[i] = data_counts[i][::-1]
    '''

    no_batches = 0
    transforms = []

    def hor_sym(matrix): return transformations.transform_matrix_channels(
        matrix, transformations.symmetric, 90)
    #transforms.append([hor_sym, hor_sym])
    #transforms.append([lambda matrix : transformations.change_brightness_contrast(matrix, 30, 50), lambda matrix : matrix])
    train_loader = utils.multi_data_loader(data_insts, 10, settings.PREFIX_DATA, settings.PREFIX_DENSITIES, data, transforms, shuffle=False)

    i = 0
    #train_insts = list(train_loader)
    for batch_insts, batch_densities, batch_counts, batch_masks in train_loader:
        batch_idx = 0
        if settings.TEMPORAL:
            for idx_temp in range(len(batch_insts[batch_idx])):
                for idx in range(len(batch_insts[batch_idx][idx_temp])):
                    #frame = data_insts[0][int(i/settings.SEQUENCE_SIZE)][idx]
                    frame = None
                    X = batch_insts[batch_idx][idx_temp][idx]
                    if settings.USE_MASK:
                        count = None
                    else:
                        count = batch_counts[batch_idx][idx_temp][idx]
                    density = batch_densities[batch_idx][idx_temp][idx]

                    if settings.USE_MASK:
                        mask = batch_masks[batch_idx][idx_temp][idx]
                    else:
                        mask = None

                    show_plot(frame, i, X, density, count, mask)

                    plt.show()
                    i += 1

        else:
            for idx in range(len(batch_insts[batch_idx])):
                #frame = data_insts[0][i]
                frame = None
                '''
                print(data_insts[batch_idx][i][1]) 
                if data_insts[batch_idx][i][1] != '20160704-12':
                    i += 1
                    continue
                '''
                X = batch_insts[batch_idx][idx]
                if settings.USE_MASK:
                    count = None
                else:
                    count = batch_counts[batch_idx][idx]
                density = batch_densities[batch_idx][idx]

                if settings.USE_MASK:
                    mask = batch_masks[batch_idx][idx]
                else:
                    mask = None

                show_plot(frame, i, X, density, count, mask)

                plt.show()
                i += 1
