import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import settings
from load_webcamt import CameraData, CameraTimeData, FrameData, VehicleData
import load_webcamt
from load_ucspeds import VideoDataUCS, FrameDataUCS
import load_ucspeds


def show_plot(cid, i, X, density, count=None, mask=None):
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
    data = load_webcamt.load_data(1, None, compute_mask=True, max_frames_per_video=10)
    #data =  load_data.load_data_from_file('first', 'first')
    i = 0
    for domain in data:
        #domain = 572
        for ct in data[domain].camera_times:
            #ct = '20160429-15'
            frame_ids = list(data[domain].camera_times[ct].frames.keys())
            frame_ids.sort(reverse=False)
            for id in frame_ids:
                #id = 300
                frame_data = data[domain].camera_times[ct].frames[id]
                if frame_data.frame is not None:
                    dict_frame = {'None': frame_data.frame}
                    dict_density = {'None': frame_data.density}
                    if settings.LOAD_DATA_AUGMENTATION:
                        dict_frame.update(frame_data.augmentation)
                        dict_density.update(frame_data.density_augmentation)
                    for aug_key in dict_frame:
                        frame = None
                        X = dict_frame[aug_key]
                        
                        cid = str(domain)+'/'+ str(ct)+'/'+str(frame_data.id)
                        count = len(frame_data.vehicles)
                        density = dict_density[aug_key]
                        mask = data[domain].camera_times[ct].mask
                        show_plot(cid, i, X, density, count, mask)
                        i += 1