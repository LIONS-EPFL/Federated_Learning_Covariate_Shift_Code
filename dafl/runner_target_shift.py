import os
from typing import Dict, List 

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.datasets.cifar import CIFAR10, CIFAR100
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
    if args.data_augmentation and args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    elif args.data_augmentation and args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
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
    elif args.dataset == 'cifar10':
        trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        trainset.data = torch.tensor(trainset.data)
        trainset.targets = torch.tensor(trainset.targets)
        testset.data = torch.tensor(testset.data)
        testset.targets = torch.tensor(testset.targets)
    elif args.dataset == 'cifar100':
        trainset = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        trainset.data = torch.tensor(trainset.data)
        trainset.targets = torch.tensor(trainset.targets)
        testset.data = torch.tensor(testset.data)
        testset.targets = torch.tensor(testset.targets)
    else:
        raise NotImplementedError

    logger.info(f"Trainset class count: {list(get_targets_counts(trainset.targets).values())}")
    logger.info(f"Testset class count: {list(get_targets_counts(testset.targets).values())}")

    if args.client_mode == 'multi':
        # Target shift
        if args.shift == 'test':
            client_label_dist_test = [{0: 0.5}, {1: 0.5}]
            client_label_dist_train = [{}, {}] # uniform
        elif args.shift == 'test-train':
            client_label_dist_test = [{0: 0.5}, {1: 0.5}]
            client_label_dist_train = [{1: 0.5}, {0: 0.5}]
        elif args.shift == 'train':
            client_label_dist_test = [{}, {}]
            client_label_dist_train = [{1: 0.5}, {0: 0.5}]
        elif args.shift == 'none':
            client_label_dist_test = [{}]
            client_label_dist_train = [{}]
        elif args.shift == 'extreme':
            min_p = 0.2
            maj_p = 1 - min_p
            client_label_dist_test = [{i: maj_p/5.0 for i in range(5)}, {i: min_p/5.0 for i in range(5)}]
            client_label_dist_train = [{i: min_p/5.0 for i in range(5)}, {i: maj_p/5.0 for i in range(5)}]
        elif args.shift == '3buckets':
            small = 0.01
            large = 1.0
            normalize = lambda d: {i:v/sum(d.values()) for i,v in d.items()}
            client0_tran = {}
            client0_tran.update({i: large for i in range(3)})
            client0_tran.update({i: small for i in range(3,7)})
            client0_tran.update({i: small for i in range(7,10)})
            client0_tran = normalize(client0_tran)
            client0_test = {}
            client0_test.update({i: small for i in range(3)})
            client0_test.update({i: large for i in range(3,7)})
            client0_test.update({i: large for i in range(7,10)})
            client0_test = normalize(client0_test)
            client1_tran = {}
            client1_tran.update({i: small for i in range(3)})
            client1_tran.update({i: small for i in range(3,7)})
            client1_tran.update({i: large for i in range(7,10)})
            client1_tran = normalize(client1_tran)
            client1_test = {}
            client1_test.update({i: small for i in range(3)})
            client1_test.update({i: large for i in range(3,7)})
            client1_test.update({i: small for i in range(7,10)})
            client1_test = normalize(client1_test)
            client_label_dist_test = [client0_test, client1_test]
            client_label_dist_train = [client0_tran, client1_tran]
        elif args.shift == '2buckets':
            small = 0.01
            large = 1.0
            normalize = lambda d: {i:v/sum(d.values()) for i,v in d.items()}
            client0_tran = {}
            client0_tran.update({i: small for i in range(5)})
            client0_tran.update({i: small for i in range(5,10)})
            client0_tran = normalize(client0_tran)
            client0_test = {}
            client0_test.update({i: small for i in range(5)})
            client0_test.update({i: large for i in range(5,10)})
            client0_test = normalize(client0_test)
            client1_tran = {}
            client1_tran.update({i: small for i in range(5)})
            client1_tran.update({i: large for i in range(5,10)})
            client1_tran = normalize(client1_tran)
            client1_test = {}
            client1_test.update({i: large for i in range(5)})
            client1_test.update({i: small for i in range(5,10)})
            client1_test = normalize(client1_test)
            client_label_dist_test = [client0_test, client1_test]
            client_label_dist_train = [client0_tran, client1_tran]
        elif args.shift == '2buckets-same':
            small = 0.05
            large = 1.0
            normalize = lambda d: {i:v/sum(d.values()) for i,v in d.items()}
            client_tran = {}
            client_tran.update({i: small for i in range(5)})
            client_tran.update({i: large for i in range(5,10)})
            client_tran = normalize(client_tran)
            client_test = {}
            client_test.update({i: large for i in range(5)})
            client_test.update({i: small for i in range(5,10)})
            client_test = normalize(client_test)
            client_label_dist_test = [client_test, client_test]
            client_label_dist_train = [client_tran, client_tran]
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
        if args.shift == '5clients':
            assert args.num_classes == 10
            client_label_dist_train = [{i: 0.95} for i in range(5,10)]
            client_label_dist_test = [{i: 0.95} for i in range(5)]
        elif args.shift == '10clients':
            assert args.num_classes == 100
            frac = 0.90/10
            client_label_dist_train = [{j: frac for j in range(i,i+5)} for i in range(50,100,5)]
            client_label_dist_test = [{j: frac for j in range(i,i+5)} for i in range(0,50,5)]
        elif args.shift == '10clients-v2':
            assert args.num_classes == 10
            client_label_dist_train = [{i: 0.95} for i in range(10)]
            client_label_dist_test = [{9-i: 0.95} for i in range(10)]
        elif args.shift == '2class-single':
            assert args.num_classes == 2
            client_label_dist_test = [{0: 0.05, 1: 0.95}]
            client_label_dist_train = [{0: 0.95, 1: 0.05}]
        elif args.shift == '2class-unitest-single':
            assert args.num_classes == 2
            client_label_dist_test = [{0: 0.5, 1: 0.5}]
            client_label_dist_train = [{0: 0.95, 1: 0.05}]

        target_shift = TargetShift(num_classes=args.num_classes)
        true_ratios = target_shift.get_ratios(client_label_dist_test, client_label_dist_train, combine_testsets=args.combine_testsets)
        trainsets = target_shift.split_dataset(trainset.data, trainset.targets, client_label_dist_train, transform=transform_train)
        testsets = target_shift.split_dataset(testset.data, testset.targets, client_label_dist_test, transform=transform_test)
        clients = []
        test_clients = []
        criterion = F.cross_entropy

        # Make client0 trainset smaller
        if args.shift == '2buckets':
            def reduce_uniformly(ds: InMemoryDataset, m):
                assert isinstance(ds, InMemoryDataset)
                size = len(ds)
                permut = np.random.permutation(size)
                ds.data, ds.targets = ds.data[permut], ds.targets[permut]
                ds.data, ds.targets = ds.data[:size//m], ds.targets[:size//m]
            reduce_uniformly(trainsets[0], m=20)

        for i in range(len(client_label_dist_test)):
            for k in range(args.num_client_duplicates):
                id_ = i*args.num_client_duplicates+k
                logger.info(f"Client {id_} trainset class count: {list(get_targets_counts(trainsets[i].targets).values())}")
                logger.info(f"Client {id_} testset class count: {list(get_targets_counts(testsets[i].targets).values())}")
                ratio_model = get_ratio_model(i, true_ratios, args)
                client_model = get_cls_model(args)
                c = Client(id_, trainsets[i], testsets[i], client_model, ratio_model, criterion, args)
                clients.append(c)
            test_clients.append(c)

        server_model = get_cls_model(args)
        server = Server(server_model, clients, criterion, args)

        def on_ratio_epoch_end(client: Client, statistics: dict):
            statistics_prefixed = {f're_client{client.id_}_{k}':v for k,v in statistics.items()}            
            wandb.log(statistics_prefixed)

        def on_step_end(step):
            if step % args.test_interval == 0:
                wandb.log({'train_step': step}, commit=False)
                server.test(test_clients)

        # Train / test
        logger.info(f"Tight ratio estimation upper bounds: {true_ratios.max(1)[0].tolist()}")
        if args.train_re:
            server.train_ratio_estimators(
                combine_testsets=args.combine_testsets,
                on_epoch_end=on_ratio_epoch_end)
        if args.num_client_duplicates == 1: # test_ratio_estimators does not support duplication
            test_ratio_estimators(server, trainset, transform_test, target_shift, 
                                    client_label_dist_test, client_label_dist_train, args)
        last_step = server.train(on_step_end=on_step_end)
        wandb.log({'train_step': last_step+1}, commit=False)
        server.test(test_clients)
    
    # Cleanup
    wandb.finish()

def get_cls_model(args):
    if args.model == "lenet":
        return LeNet(rep_dim=args.num_classes).to(args.device)
    elif args.model == "resnet18":
        return ResNet18(num_classes=args.num_classes).to(args.device)
    else:
        raise NotImplemented("Only LeNet and ResNet18 model type is supported")


def get_ratio_model(client_id, true_ratios, args):
    if args.use_true_ratio:
        assert not args.train_re, "True ratio model cannot train"

        # Note that TrueRatioModel requires the target to be passed
        true_ratios = true_ratios[client_id]
        return TrueRatioModel(true_ratios)

    if args.train_re:
        return LeNet(rep_dim=1, force_pos=args.force_pos).to(args.device) 
    else:
        return UniformRatioModel()


def test_ratio_estimators(server, trainset, transform_test, target_shift, 
                client_label_dist_test, client_label_dist_train, args):
    """Checks target conditional ratios for all clients. 
    Preferably set args.test_batch_size large since only one batch is used per class.
    """
    test_probs = target_shift.get_probs(client_label_dist_test)
    train_probs = target_shift.get_probs(client_label_dist_train)

    train_target_dataloaders = []
    for i in range(target_shift.num_classes):    
        idx = trainset.targets == i
        data, targets = trainset.data[idx], trainset.targets[idx]
        dataset = InMemoryDataset(data, targets, transform=transform_test)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size, 
                                                shuffle=True, **args.dataloader_kwargs)
        train_target_dataloaders.append(dataloader)

    for i in range(len(server.clients)):
        i_test_probs = test_probs[i]/torch.sum(test_probs[i])
        i_train_probs = train_probs[i]/torch.sum(train_probs[i])
        i_true_ratio = i_test_probs/i_train_probs
        logger.info(f"Client {i} true target conditional ratio: {i_true_ratio}")
        predicted_target_conditional_ratio = []
        for j in range(target_shift.num_classes):
            dataloader = train_target_dataloaders[j]
            img, target = next(iter(dataloader))
            img = img.to(args.device)
            if args.use_true_ratio:
                mean_ratio = server.clients[i].ratio_model(img, target).mean()
            else:
                mean_ratio = server.clients[i].ratio_model(img).mean()
            predicted_target_conditional_ratio.append(mean_ratio.item())
        logger.info(f"Client {i} predicted target conditional ratio: {predicted_target_conditional_ratio}")


if __name__ == "__main__":
    main()
