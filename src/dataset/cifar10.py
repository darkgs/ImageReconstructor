
import os
import pickle

import numpy as np
import torch

from utils.utils import *

class Cifar10(object):
    def __init__(self, path_cifar10):
        assert(os.path.exists(path_cifar10))

        self._path_cifar10 = path_cifar10

        #self.data_shape = (32, 32, 3)
        self.classes = 10
        self.name = "cifar-10"

    @lazy_attr
    def data(self):
        data = {}
        def merge_cifar10_files(files):
            images = []
            labels = []
            for file_path in files:
                with open(file_path, 'rb') as f_train:
                    # 32 x 32 x 3 images
                    dict_data = pickle.load(f_train, encoding='bytes')

                images.append(dict_data[b'data'])
                labels.append(dict_data[b'labels'])

            #images = np.reshape(np.stack(images, axis=0), (-1, *self.data_shape)).astype('float32')
            images = np.reshape(np.stack(images, axis=0), (-1, 3, 32, 32)).astype('float32')
            #images = np.transpose(images, (0,2,3,1))
            labels = np.reshape(np.stack(labels, axis=0), (-1,)).astype('int32')

            return images, labels

        train_files = [self._path_cifar10 + "/data_batch_" + str(i+1) for i in range(5)]
        test_files = ["{}/test_batch".format(self._path_cifar10)]

        train_images, train_labels = merge_cifar10_files(train_files)
        test_images, test_labels = merge_cifar10_files(test_files)

        # dividing datasets
        num_train = int(train_images.shape[0] * 0.8)

        data['train'] = {
            'images': train_images[:num_train,],
            'labels': train_labels[:num_train,],
        }

        data['valid'] = {
            'images': train_images[num_train:,],
            'labels': train_labels[num_train:,],
        }

        data['test'] = {
            'images': test_images,
            'labels': test_labels,
        }
        return data


class Cifar10Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, cifar10_data):
        self._data = cifar10_data
        self._data_len = cifar10_data['images'].shape[0]

    def __getitem__(self, idx):
        return (self._data['images'][idx] / 255., self._data['labels'][idx])

    def __len__(self):
        return self._data_len

    def sample_images(self, num_sample):
        sample_idxs = np.random.permutation(len(self._data['images']))[:num_sample]
        return self._data['images'][sample_idxs] / 255.


def get_cifar10_torch_datasets(path_cifar10):
    dict_datasets = {
        'data': {},
        'classes': 10,
    }
    cifar10 = Cifar10(path_cifar10)

    for data_type in ['train', 'valid', 'test']:
        dict_datasets['data'][data_type] = Cifar10Dataset(cifar10.data[data_type])

    return dict_datasets


