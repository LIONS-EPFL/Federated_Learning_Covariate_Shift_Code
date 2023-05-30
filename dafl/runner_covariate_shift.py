import os
from typing import Dict, List 

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision import transforms
from dafl.args import Args
from dafl.client import Client
from dafl.colored_mnist import ColoredMNIST, ColoredMNISTTrueRatioModel
from dafl.logger import getLogger
from dafl.mnist_lenet import ConvNet_3Channel_small
from dafl.resnet import ResNet18, ResNet7
from dafl.server import Server
from dafl.ratio_estimation import TrueRatioModel, UniformRatioModel
from dafl.target_shift import InMemoryDataset, TargetShift, get_targets_counts
import wandb


from pathlib import Path
from datargs import parse


logger = getLogger(__name__)


def main():
    # Parse arguments
    args = parse(Args)
    logger.info(args)

    # Logging
    wandb.init(
        tags=args.wandb_tags,
        project=args.wandb_project, 
        entity=args.wandb_entity, 
        name=args.wandb_name, 
        id=args.wandb_id,
        config=args)

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

    target_transform = lambda target: float(target)

    trainsets = [
        ColoredMNIST(root='./data', env='train1', transform=transform_train, target_transform=target_transform, grayscale=args.force_grayscale),
        ColoredMNIST(root='./data', env='train2', transform=transform_train, target_transform=target_transform, grayscale=args.force_grayscale),
    ]
    testsets = [
        ColoredMNIST(root='./data', env='test2', transform=transform_test, target_transform=target_transform, grayscale=args.force_grayscale),
        ColoredMNIST(root='./data', env='test1', transform=transform_test, target_transform=target_transform, grayscale=args.force_grayscale),
    ]
    probs = [
        {'train_flip_prob': 0.5,
         'test_flip_prob': 0.2},
        {'train_flip_prob': 0.2,
         'test_flip_prob': 0.8}
    ]
    clients = []
    criterion = F.binary_cross_entropy_with_logits

    if args.combine_testsets:
        total = sum(p['test_flip_prob'] for p in probs)
        probs[0]['test_flip_prob'] = total / 2
        probs[1]['test_flip_prob'] = total / 2

    if args.client_mode == 'single':
        trainsets = [trainsets[1]]
        testsets = [testsets[1]]
        probs = [probs[1]]

    def flatten(model):
        return torch.nn.Sequential(model, torch.nn.Flatten(0)).to(args.device)

    for i in range(len(trainsets)):
        # logger.info(f"Client {i} trainset class count: {list(get_targets_counts(trainsets[i].targets).values())}")
        # logger.info(f"Client {i} testset class count: {list(get_targets_counts(testsets[i].targets).values())}")
        ratio_model = get_ratio_model(probs[i], args)
        client_model = flatten(get_model(args, rep_dim=1))
        c = Client(i, trainsets[i], testsets[i], client_model, ratio_model, criterion, args)
        clients.append(c)

    server_model = flatten(get_model(args, rep_dim=1))
    server = Server(server_model, clients, criterion, args)

    # Since binary classification tasks
    server.pred = lambda output: torch.round(torch.sigmoid(output))
    for client in clients:
        client.pred = lambda output: torch.round(torch.sigmoid(output))

    def on_epoch_end(client: Client, statistics: dict):
        statistics_prefixed = {f're_client{client.id_}_{k}':v for k,v in statistics.items()}
        wandb.log(statistics_prefixed)

    # Train / test
    if args.train_re:
        server.train_ratio_estimators(
            combine_testsets=args.combine_testsets,
            on_epoch_end=on_epoch_end)
    server.train()
    server.test()

    # Cleanup
    wandb.finish()


def get_model(args, rep_dim=1):
    if args.model == 'resnet7':
        model = ResNet7(num_classes=rep_dim).to(args.device)
    elif args.model == 'lenet':
        model = ConvNet_3Channel_small(rep_dim=rep_dim).to(args.device)
    return model


def get_ratio_model(probs, args):
    if args.use_true_ratio:
        assert not args.train_re, "True ratio model cannot train"

        return ColoredMNISTTrueRatioModel(
            train_flip_prob=probs['train_flip_prob'],
            test_flip_prob=probs['test_flip_prob'],
        ).to(args.device)
    if args.train_re:
        return get_model(args, rep_dim=1)
    else:
        return UniformRatioModel()


if __name__ == "__main__":
    main()
