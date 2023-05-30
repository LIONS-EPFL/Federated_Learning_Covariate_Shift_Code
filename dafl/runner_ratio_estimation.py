import os
from typing import Dict, List 

import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision import transforms
from dafl.args import Args
from dafl.cifar10_lenet import CIFAR10_LeNet
from dafl.client import Client
from dafl.logger import getLogger
from dafl.mnist_lenet import D3RE_LeNet, LeNet
from dafl.pretrained_classifier import LabelBasedRatio, LinearOnFeatures, pretrained_mnist_model
from dafl.resnet import ResNet18, ResNet7
from dafl.server import Server
from dafl.ratio_estimation import RatioEstimation, UniformRatioModel
from dafl.ratio_estimation_d3re import RatioEstimation as D3RERatioEstimation
from dafl.target_shift import CombinedInMemoryDataset, InMemoryDataset, TargetShift, get_targets_counts
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
    if args.dataset == 'mnist':
        trainset = MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = MNIST(root='./data', train=False, download=True, transform=transform_test)
    elif args.dataset == 'fmnist':
        trainset = FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise NotImplementedError

    logger.info(f"Trainset class count: {list(get_targets_counts(trainset.targets).values())}")
    logger.info(f"Testset class count: {list(get_targets_counts(testset.targets).values())}")

    if args.shift == '2buckets-single':
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
    elif args.shift == '2class':
        client_label_dist_test = [{0: 0.05, 1: 0.95}]
        client_label_dist_train = [{0: 0.95, 1: 0.05}]
    elif args.shift == '2class-unitest':
        client_label_dist_test = [{0: 0.5, 1: 0.5}]
        client_label_dist_train = [{0: 0.95, 1: 0.05}]
    else:
        client_label_dist_test = [{0: 0.5}]
        client_label_dist_train = [{}]

    ls = TargetShift(num_classes=args.num_classes)
    trainsets = ls.split_dataset(trainset.data, trainset.targets, client_label_dist_train, transform=transform_train)
    testsets = ls.split_dataset(testset.data, testset.targets, client_label_dist_test, transform=transform_test)
    i = 0

    # Simulate case where only few (`public_testset_size`) examples are available
    # testset_size = len(testsets[i])
    # public_testset, _ = torch.utils.data.random_split(testsets[i], [args.public_testset_size, testset_size - args.public_testset_size])
    public_testset = testsets[i]

    test_target_dataloaders = []
    for j in range(ls.num_classes):    
        idx = public_testset.targets == j
        data, targets = public_testset.data[idx], public_testset.targets[idx]
        dataset = InMemoryDataset(data, targets, transform=transform_test)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size, 
                                                shuffle=True, **args.dataloader_kwargs)
        test_target_dataloaders.append(dataloader)

    train_target_dataloaders = []
    for j in range(ls.num_classes):    
        idx = trainsets[i].targets == j
        data, targets = trainsets[i].data[idx], trainsets[i].targets[idx]
        dataset = InMemoryDataset(data, targets, transform=transform_test)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size, 
                                                shuffle=True, **args.dataloader_kwargs)
        train_target_dataloaders.append(dataloader)

    def on_epoch_end(statistics: dict):
        ratio_model.eval()

        # Check target conditional ratios
        test_probs = ls.get_probs(client_label_dist_test)
        test_probs = test_probs/torch.sum(test_probs)
        train_probs = ls.get_probs(client_label_dist_train)
        train_probs = train_probs/torch.sum(train_probs)
        true_ratio = test_probs/train_probs
        logger.info(f"True target conditional ratio: {true_ratio}")
        
        loaders = {
            'test': test_target_dataloaders,
            'train': train_target_dataloaders,
        }
        with torch.no_grad():
            for k, dataloaders in loaders.items():
                predicted_target_conditional_ratio = []
                for i in range(ls.num_classes):
                    dataloader = dataloaders[i]
                    img, target = next(iter(dataloader))
                    img = img.to(args.device)
                    if args.model == 'label-based':
                        mean_ratio = ratio_model(target).mean()
                    else:
                        mean_ratio = ratio_model(img).mean()
                    predicted_target_conditional_ratio.append(mean_ratio.item())
                logger.info(f"Predicted target conditional ratio on {k}: {predicted_target_conditional_ratio}")
                
                wandb.log({
                    f'{k}_cond_ratio_max': max(predicted_target_conditional_ratio), 
                    f'{k}_cond_ratio_min': min(predicted_target_conditional_ratio),
                    f'{k}_cond_ratio_mse': ((torch.tensor(predicted_target_conditional_ratio) - true_ratio)**2).mean(),
                }, commit=False)

        wandb.log(statistics, commit=True)
        ratio_model.train()

    logger.info(f"Target-shifted trainset class count: {list(get_targets_counts(trainsets[i].targets).values())}")
    logger.info(f"Target-shifted testset class count before reduction: {list(get_targets_counts(testsets[i].targets).values())}")
    if args.model == 'lenet':
        ratio_model = LeNet(rep_dim=1, force_pos=args.force_pos).to(args.device)
    elif args.model == 'd3re-lenet':
        ratio_model = D3RE_LeNet(rep_dim=1, force_pos=args.force_pos).to(args.device)
    elif args.model == 'resnet7':
        ratio_model = ResNet7(num_classes=1, ch_in=1).to(args.device)
    elif args.model == 'pretrained':
        n_hiddens = [256, 256]
        pretrained_clf = pretrained_mnist_model(pretrained='./classifiers/mnist_classifier_28.pt', n_hiddens=n_hiddens)
        feature_extractor = torch.nn.Sequential(
            # Renormalize
            # torchvision.transforms.Normalize((-0.5,), (1/0.5,)), 
            torch.nn.Flatten(),
            *list(pretrained_clf.layers.values())[:-1]
        )
        ratio_model = LinearOnFeatures(feature_extractor, num_features=n_hiddens[-1]).to(args.device)
    elif args.model == 'label-based':
        ratio_model = LabelBasedRatio(num_classes=args.num_classes).to(args.device)

    if args.d3re_impl:
        dataset = CombinedInMemoryDataset(trainsets[i], public_testset)
        re = D3RERatioEstimation(i, ratio_model, dataset, args)
    else:
        re = RatioEstimation(i, ratio_model, trainsets[i], public_testset, args)
    
    re.train(on_epoch_end=on_epoch_end)

    # Cleanup
    wandb.finish()


if __name__ == "__main__":
    main()
