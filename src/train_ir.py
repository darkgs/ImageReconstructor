
import os, sys
import signal

import argparse

import numpy as np

import torch
import torch.nn as nn
import torchvision

from dataset.cifar10 import get_cifar10_torch_datasets

from utils.utils import weights_init
from utils.utils import mkdir

class SimpleVAE(nn.Module):
    def __init__(self):
        super(SimpleVAE, self).__init__()

        self._encode = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self._linear = nn.Sequential(
            nn.Linear(6144, 1024),
            nn.ReLU(),
            nn.Linear(1024, 6144),
            nn.ReLU(),
        )

        self._decode = nn.Sequential(
            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        self.apply(weights_init)
    
    def forward(self, x):
        x = self._encode(x)
        shape_before = x.shape

        x = x.view(x.size(0), -1)
        x = self._linear(x)
        x = x.view(x.size(0), *shape_before[1:])

        x = self._decode(x)
        return x

class ImageReconstructor(object):
    def __init__(self, datasets, model, path_saved_model):

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._dataloaders = {}
        for data_type, batch_size, shuffle in [
                ('train', 256, True), ('valid', 256, False), ('test', 64, False)]:
            self._dataloaders[data_type] = torch.utils.data.DataLoader(datasets['data'][data_type],
                batch_size=batch_size, shuffle=shuffle, num_workers=16)

        self._model = model.to(self._device)

        self._criterion = nn.MSELoss()
        self._optimizer = torch.optim.Adam(self._model.parameters())

        self._path_saved_model = path_saved_model
        mkdir(os.path.dirname(self._path_saved_model))

        def gracefull_die(sig, fram):
            print("Trigger the gracefull die")
            if os.path.exists(self._path_saved_model):
                os.system("touch {}".format(self._path_saved_model))
            sys.exit(0)

        signal.signal(signal.SIGINT, gracefull_die)

    def save(self, epoch, valid_acc):
        dict_saved_data = {
            'epoch': epoch,
            'valid_acc': valid_acc,
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

        torch.save(dict_saved_data, self._path_saved_model)


    def load(self):
        if not os.path.exists(self._path_saved_model):
            return 1, -1

        dict_saved_data = torch.load(self._path_saved_model)

        try:
            self._model.load_state_dict(dict_saved_data['model'])
            self._optimizer.load_state_dict(dict_saved_data['optimizer'])
        except:
            print("model unmatch!!")
            os.system("rm -rf {}".format(self._path_saved_model))
            return 1, -1

        print("Load a model at epoch {} with valid acc {:.4f}".format(
                dict_saved_data['epoch'], dict_saved_data['valid_acc']))
        return dict_saved_data['epoch'], dict_saved_data['valid_acc']

    def train_a_epoch(self):
        self._model.train()

        loss_sum = 0
        loss_count = 0
        for x, _ in self._dataloaders['train']:
            self._optimizer.zero_grad()

            x = x.to(self._device)
            x_recon = self._model(x)

            loss = self._criterion(x, x_recon)
            loss.backward()

            self._optimizer.step()

            loss_sum += loss.item()
            loss_count += x.size(0)

        return loss_sum / loss_count

    def test(self, data_type):
        assert(data_type in ['train', 'valid', 'test'])

        self._model.eval()

        loss_sum = 0
        loss_count = 0
        for x, _ in self._dataloaders[data_type]:
            x = x.to(self._device)
            x_recon = self._model(x)

            err = (x - x_recon).view(x.size(0), -1)
            print(err.shape)
            break

        return loss_sum / loss_count

    def train(self):
        epoch_st, top_valid_acc = self.load()

        epochs = 100000
        prev_valid_acc = -1
        for epoch in range(epoch_st, epochs+1):
            epoch_train_loss = self.train_a_epoch()
            epoch_valid_acc = self.test('valid')

            print("{:.4f}".format(epoch_train_loss))
            
            continue

            epoch_log = "Epoch {}: train loss({:.4f}), valid acc ({:.4f})".format(
                epoch, epoch_train_loss, epoch_valid_acc)

            if top_valid_acc <= epoch_valid_acc:
                top_valid_acc = epoch_valid_acc
                self.save(epoch, top_valid_acc)
                epoch_log += ": Saved! - valid acc({:.4f})".format(top_valid_acc)

            print(epoch_log)


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Train a model F')
    parser.add_argument('--path_cifar10', type=str, default="data/cifar-10", \
            help='path for cifar10')
    parser.add_argument('--path_saved_model', type=str, default=None, \
            help='path where the weights of model be stored')
    args = parser.parse_args()

    # parse from args
    path_cifar10 = args.path_cifar10
    path_saved_model = args.path_saved_model

    # Load the target dataset, model
    datasets = get_cifar10_torch_datasets(path_cifar10)
    model = SimpleVAE()

    # Generate a ImageReconstructor
    ir = ImageReconstructor(datasets, model, path_saved_model)
    ir.train()


if __name__ == '__main__':
    main()
