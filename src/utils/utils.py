
def lazy_attr(fn):
    '''
        Decorator that makes a property lazy-evaluated.
    '''
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_attr(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_attr

def mkdir(path):
    import os
    
    if len(path) == 0 or os.path.exists(path):
        return

    os.system("mkdir -p {}".format(path))

def save_images(np_images, path_to_save, do_clip=True):
    import os

    import numpy as np

    import glob
    import matplotlib.pyplot as plt
    import imageio

    # clip values (optional)
    if do_clip:
        np_images = np.clip(np_images, 0., 1.)

    # decide the shape of image grid
    num_images = np_images.shape[0]
    num_row = 4
    num_col = int(num_images / num_row) + (1 if num_images % num_row != 0 else 0)

    # draw figure
    np_images = np.transpose(np_images, (0, 3, 2, 1))
    fig = plt.figure(figsize=(num_col, num_row))
    for i in range(num_images):
        plt.subplot(num_col, num_row, i + 1)
        plt.imshow(np_images[i,:,:,:], interpolation='bicubic')
        plt.axis('off')

    mkdir(os.path.dirname(path_to_save))

    plt.savefig(path_to_save)
    plt.close(fig)

def weights_init(m):
    import torch.nn as nn

    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

