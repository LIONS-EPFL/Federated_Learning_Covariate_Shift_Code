import os
from typing import Dict, List 

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision import transforms
from dafl.args import Args
from dafl.cifar10_lenet import CIFAR10_LeNet
from dafl.client import Client
from dafl.logger import getLogger
from dafl.mnist_lenet import LeNet
from dafl.resnet import ResNet18
from dafl.server import Server
from dafl.ratio_estimation import TrueRatioModel, UniformRatioModel
from dafl.target_shift import InMemoryDataset, TargetShift, get_targets_counts


from pathlib import Path
from datargs import parse


logger = getLogger(__name__)


def main():
    # Parse arguments
    args = parse(Args)
    logger.info(args)

    # Folder and seed
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Setup dataset
    if args.data_augmentation:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if args.dataset == 'mnist':
        trainset = MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = MNIST(root='./data', train=False, download=True, transform=transform_test)
    elif args.dataset == 'fmnist':
        trainset = FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise NotImplementedError

    #if args.shift == '2buckets-single':
    small = 0.05
    large = 1.0
    normalize = lambda d: {i:v/sum(d.values()) for i,v in d.items()}
    client1_tran = {}
    client1_tran.update({i: small for i in range(5)})
    client1_tran.update({i: large for i in range(5,10)})
    client1_tran = normalize(client1_tran)
    client1_test = {}
    client1_test.update({i: large for i in range(5)})
    client1_test.update({i: small for i in range(5,10)})
    client1_test = normalize(client1_test)
    client_label_dist_test = [client1_test]
    client_label_dist_train = [client1_tran]

    target_shift = TargetShift(num_classes=args.num_classes)
    true_ratios = target_shift.get_ratios(client_label_dist_test, client_label_dist_train, combine_testsets=args.combine_testsets)
    trainset, = target_shift.split_dataset(trainset.data, trainset.targets, client_label_dist_train, transform=transform_train)
    testset, = target_shift.split_dataset(testset.data, testset.targets, client_label_dist_test, transform=transform_test)

    logger.info(f"Trainset class count: {list(get_targets_counts(trainset.targets).values())}")
    logger.info(f"Testset class count: {list(get_targets_counts(testset.targets).values())}")
    logger.info(f"Testset class count: {true_ratios.tolist()}")

    # Optional: learn feature representation
    from sklearn.cluster import KMeans
    data = np.concatenate([trainset.data, testset.data])
    data = data.reshape(data.shape[0], -1)
    kmeans = KMeans(n_clusters=args.num_clusters).fit(data)
    train_assigns = kmeans.predict(trainset.data.reshape(trainset.data.shape[0], -1))
    test_assigns = kmeans.predict(testset.data.reshape(testset.data.shape[0], -1))
    
    uniq, counts = torch.unique(torch.tensor(train_assigns), return_counts=True)
    train_counts = dict(zip(uniq.tolist(), counts.tolist()))

    uniq, counts = torch.unique(torch.tensor(test_assigns), return_counts=True)
    test_counts = dict(zip(uniq.tolist(), counts.tolist()))

    ratios = map(lambda k: (test_counts[k]/len(testset.data)) / (train_counts[k] / len(trainset.data)), test_counts.keys())
    max_ratio = max(ratios)
    print("Max ratio:", max_ratio)

if __name__ == "__main__":
    main()
