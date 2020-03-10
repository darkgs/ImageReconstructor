
import os, sys
import signal

import argparse

import numpy as np

import torch
import torch.nn as nn
import torchvision

from dataset.cifar10 import get_cifar10_torch_datasets

from model.alexnet import AlexNet
from model.googlenet import GoogLeNet

from utils.utils import weights_init
from utils.utils import mkdir

class ImageClassifier(object):
    def __init__(self, datasets, model_f, path_saved_model):

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._dataloaders = {}
        for data_type, batch_size, shuffle in [
                ('train', 256, True), ('valid', 256, False), ('test', 64, False)]:
            self._dataloaders[data_type] = torch.utils.data.DataLoader(datasets['data'][data_type],
                batch_size=batch_size, shuffle=shuffle, num_workers=16)

        self._model_f = model_f.to(self._device)

        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.Adam(self._model_f.parameters())

        self._classes = datasets['classes']

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
            'model_f': self._model_f.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

        torch.save(dict_saved_data, self._path_saved_model)


    def load(self):
        if not os.path.exists(self._path_saved_model):
            return 1, -1

        dict_saved_data = torch.load(self._path_saved_model)

        try:
            self._model_f.load_state_dict(dict_saved_data['model_f'])
            self._optimizer.load_state_dict(dict_saved_data['optimizer'])
        except:
            print("model unmatch!!")
            os.system("rm -rf {}".format(self._path_saved_model))
            return 1, -1

        print("Load a model at epoch {} with valid acc {:.4f}".format(
                dict_saved_data['epoch'], dict_saved_data['valid_acc']))
        return dict_saved_data['epoch'], dict_saved_data['valid_acc']

    def train_a_epoch(self):
        self._model_f.train()

        loss_sum = 0
        loss_count = 0
        for x, y in self._dataloaders['train']:
            self._optimizer.zero_grad()

            x = x.to(self._device)
            y = y.type(torch.LongTensor).to(self._device)

            y_pred = self._model_f(x)

            loss = self._criterion(y_pred, y)
            loss.backward()

            self._optimizer.step()

            loss_sum += loss.item()
            loss_count += y.size(0)

        return loss_sum / loss_count

    def test(self, data_type):
        self._model_f.eval()

        acc_sum = 0
        acc_count = 0
        for x, y in self._dataloaders[data_type]:
            x = x.to(self._device)
            y = y.type(torch.LongTensor).to(self._device)

            y_pred = self._model_f(x)
            _, y_pred = torch.max(y_pred.data, 1)

            acc_sum += (y_pred == y).sum().item()
            acc_count += y.size(0)

        return acc_sum / acc_count

    def train(self):
        epoch_st, top_valid_acc = self.load()

        epochs = 100000
        prev_valid_acc = -1
        for epoch in range(epoch_st, epochs+1):
            epoch_train_loss = self.train_a_epoch()
            epoch_valid_acc = self.test('valid')

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
    name_model_f = args.model_f
    path_saved_model = args.path_saved_model

    # Load the target dataset
    datasets = get_cifar10_torch_datasets(path_cifar10)

    print("hello")

    return

    # Generate a torch image classifier
    ic = ImageClassifier(datasets, model_f, path_saved_model)
    ic.train()

    ic.load()
    train_acc = ic.test('train')
    valid_acc = ic.test('valid')
    test_acc = ic.test('test')

    print('train_acc({:.4f}) valid acc({:.4f}) test acc({:.4f})'.format(train_acc, valid_acc, test_acc))


if __name__ == '__main__':
    main()
