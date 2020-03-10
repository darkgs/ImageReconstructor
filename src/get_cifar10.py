
import os

import argparse

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Download cifar10 dataset')
    parser.add_argument('--path_data', type=str, default="data", help='path of data')
    parser.add_argument('--path_cifar10', type=str, default="data/cifar-10", \
            help='path for cifar10')
    args = parser.parse_args()

    # parse from args
    path_data = args.path_data
    path_cifar10 = args.path_cifar10
    path_cifar10_tar = "{}/cifar-10-python.tar.gz".format(path_data)

    if os.path.exists(path_cifar10):
        os.system("rm -f {}".format(path_cifar10))

    # Download and Extract the cifar-10
    os.system("wget -O {} https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz".format(path_cifar10_tar))
    os.system("tar xvzf {} -C {}".format(path_cifar10_tar, path_data))
    os.system("mv -f {}/cifar-10-batches-py {}".format(path_data, path_cifar10))
    os.system("rm -f {}".format(path_cifar10_tar))
    os.system("touch {}".format(path_cifar10))


if __name__ == "__main__":
    main()

