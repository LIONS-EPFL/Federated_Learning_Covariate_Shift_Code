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
from dafl.colored_mnist import ColoredMNIST


from pathlib import Path
from datargs import parse


dataset_name = "Colored MNIST"


# Setup dataset
transform_train = transforms.Compose([
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
transforms.ToTensor(),
])
target_transform = lambda target: float(target)

# Set divide=1 such that client1 train is not reduced. Otherwise n<<d.
trainsets = [
    ColoredMNIST(root='./data', env='train1', transform=transform_train, target_transform=target_transform, grayscale=False, divide=1),
    ColoredMNIST(root='./data', env='train2', transform=transform_train, target_transform=target_transform, grayscale=False, divide=1),
]
testsets = [
    ColoredMNIST(root='./data', env='test2', transform=transform_test, target_transform=target_transform, grayscale=False, divide=1),
    ColoredMNIST(root='./data', env='test1', transform=transform_test, target_transform=target_transform, grayscale=False, divide=1),
]

for set in trainsets:
    set.data = torch.stack([data[0] for data in set])

#trainsets[0].data = trainsets[0].data.repeat(5, 1, 1, 1)

for set in testsets:
    set.data = torch.stack([data[0] for data in set])

# Plot test/train ratio for each client
for i in range(len(trainsets)):
    d_tr = trainsets[i].data
    d_tr = d_tr.reshape(d_tr.shape[0], -1)
    _,S_tr,_ = torch.linalg.svd(d_tr)

    d_te = testsets[i].data
    d_te = d_te.reshape(d_te.shape[0], -1)
    _,S_te,_ = torch.linalg.svd(d_te)

    plt.plot(S_te/S_tr, label=f"Client {i+1}")

plt.ylabel("$\sqrt{{\\lambda'_i/\\lambda_i}}$")
plt.xlabel("$i$")
plt.legend()
plt.title(dataset_name)
plt.savefig(dataset_name, dpi=300)
plt.clf()



# %%
plt.plot(S_te, label="$\\lambda'_i$")
plt.plot(S_tr, label="$\\lambda_i$")
plt.ylabel("i")
plt.yscale("log")
plt.legend()
plt.title(dataset_name)
plt.savefig(dataset_name + "_eig", dpi=300)

# %%
