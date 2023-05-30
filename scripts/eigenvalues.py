#%%
import os
from typing import Dict, List
from attr import dataclass 

from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision import transforms
from dafl.args import Args
from dafl.cifar10_lenet import CIFAR10_LeNet
from dafl.client import Client
from dafl.logger import getLogger
from dafl.mnist_lenet import D3RE_LeNet, LeNet
from dafl.resnet import ResNet18
from dafl.server import Server
from dafl.ratio_estimation import RatioEstimation, UniformRatioModel
from dafl.target_shift import InMemoryDataset, TargetShift, get_targets_counts
import wandb


from pathlib import Path
from datargs import parse

datasets = {
    #"MNIST": MNIST, 
    "Fashion MNIST": FashionMNIST, 
    #"CIFAR10": CIFAR10,
}

for dataset_name, Dataset in datasets.items():

    transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
    trainset = Dataset(root='../../data', train=True, download=True, transform=
    transform_test)
    testset = Dataset(root='../../data', train=False, download=True, transform=transform_test)

    if dataset_name == "CIFAR10":
        trainset.data = torch.tensor(trainset.data)
        trainset.targets = torch.tensor(trainset.targets)
        testset.data = torch.tensor(testset.data)
        testset.targets = torch.tensor(testset.targets)

    client_label_dist_train = [{i: 0.95} for i in range(5,10)]
    client_label_dist_test = [{i: 0.95} for i in range(5)]

    ls = TargetShift(num_classes=10)
    trainsets = ls.split_dataset(trainset.data, trainset.targets, client_label_dist_train, transform=transform_test)
    testsets = ls.split_dataset(testset.data, testset.targets, client_label_dist_test, transform=transform_test)
    
    # Plot test/train ratio for each client
    for i in range(len(trainsets)):
        d_tr = trainsets[i].data / 255
        d_tr = d_tr.reshape(d_tr.shape[0], -1)
        _,S_tr,_ = torch.linalg.svd(d_tr)

        d_te = testsets[i].data / 255
        d_te = d_te.reshape(d_te.shape[0], -1)
        _,S_te,_ = torch.linalg.svd(d_te)

        plt.plot(S_te/S_tr, label=f"Client {i+1}")

    plt.ylabel("$\sqrt{{\\lambda'_i/\\lambda_i}}$")
    plt.xlabel("$i$")
    plt.legend()
    plt.title(dataset_name)
    plt.savefig(dataset_name, dpi=300)
    plt.clf()

    # # Plot joint_test/private_train ratio for each client
    # joint_testset = torch.utils.data.ConcatDataset(testsets)
    # joint_testset_data = torch.stack([data[0] for data in joint_testset])

    # for i in range(len(trainsets)):
    #     d_tr = trainsets[i].data / 255
    #     d_tr = d_tr.reshape(d_tr.shape[0], -1)
    #     _,S_tr,_ = torch.linalg.svd(d_tr)

    #     d_te = joint_testset_data
    #     d_te = d_te.reshape(d_te.shape[0], -1)
    #     _,S_te,_ = torch.linalg.svd(d_te)

    #     plt.plot(S_te/S_tr, label=f"Client {i+1}")

    # plt.ylabel("$\sqrt{{\\lambda'_i/\\lambda_i}}$")
    # plt.xlabel("$i$")
    # plt.legend()
    # plt.title(dataset_name)
    # plt.savefig(f"{dataset_name}_joint_test", dpi=300)
    # plt.clf()


# %%
import itertools

fig, ax = plt.subplots()
colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

for i in range(len(trainsets)):
    d_tr = trainsets[i].data / 255
    d_tr = d_tr.reshape(d_tr.shape[0], -1)
    _,S_tr,_ = torch.linalg.svd(d_tr)

    d_te = testsets[i].data / 255
    d_te = d_te.reshape(d_te.shape[0], -1)
    _,S_te,_ = torch.linalg.svd(d_te)

    color =  next(colors)
    plt.plot(S_te, label=f"Client {i+1} ($\\lambda'_i$)", color=color)
    plt.plot(S_tr, label=f"Client {i+1} ($\\lambda_i$)", color=color, linestyle="dashed")
    plt.yscale("log")

plt.ylabel("i")
plt.legend()
plt.title(dataset_name)
plt.savefig(dataset_name + "_eig", dpi=300)

# %%
