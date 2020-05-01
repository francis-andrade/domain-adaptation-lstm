from load_data import load_data
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import settings

if __name__ == '__main__':
    data = load_data(1, settings.IMAGE_NEW_SHAPE)
    i = 0
    for domain in data:
        for ct in data[domain].camera_times:
            for id in data[domain].camera_times[ct].frames:
                if data[domain].camera_times[ct].frames[id].frame is not None:
                    X = data[domain].camera_times[ct].frames[id].frame
                    X = X.reshape(X.shape[1], X.shape[2], 3)
                    cid = str(domain)+'/'+ str(ct)+'/'+str(data[domain].camera_times[ct].frames[id].id)
                    count = len(data[domain].camera_times[ct].frames[id].vehicles)
                    density = data[domain].camera_times[ct].frames[id].density
                    density = density.reshape(density.shape[1], density.shape[2], 1)
                    print('Image {}: cid={}, count={}, density_sum={:.3f}'.format(i, cid, count, np.sum(density)))
                    gs = gridspec.GridSpec(2, 2)
                    fig = plt.figure()
                    ax1 = fig.add_subplot(gs[0, 0])
                    ax1.imshow(X/255.)
                    ax1.set_title('Masked image')
                    ax2 = fig.add_subplot(gs[0, 1])
                    density = density.squeeze()
                    ax2.imshow(density, cmap='gray')
                    ax2.set_title('Density map')
                    ax3 = fig.add_subplot(gs[1, :])
                    Xh = np.tile(np.mean(X, axis=2, keepdims=True), (1, 1, 3))
                    Xh[:, :, 1] *= (1-density/np.max(density))
                    Xh[:, :, 2] *= (1-density/np.max(density))
                    ax3.imshow(Xh.astype('uint8'))
                    ax3.set_title('Highlighted vehicles')
                    plt.show()
                    i += 1