from typing import Any, Dict, List, Tuple 
from PIL import Image

import torch
import numpy as np
from torch.utils.data.dataset import Dataset


ClassId = int
Prob = float
TargetShiftConfig = Dict[ClassId, Prob]


class CombinedInMemoryDataset(Dataset):
    def __init__(self, dataset1, dataset2) -> None:
        super().__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset1_size = len(self.dataset1)
        self.dataset2_size = len(self.dataset2)

    def __len__(self):
        return self.dataset1_size + self.dataset2_size

    def __getitem__(self, index) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index) where target is index of the target class.
        """
        if index < self.dataset1_size:
            semi_target = 1
            return self.dataset1[index] + (semi_target, index)
        else:
            semi_target = 0
            index -= self.dataset1_size
            return self.dataset2[index] + (semi_target, index)


class InMemoryDataset(Dataset):
    def __init__(self, data, targets, transform=None, imagedata=True) -> None:
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.imagedata = imagedata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.imagedata:
            img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        return img, target



class TargetShift(object):
    def __init__(self, num_classes=10):
        self.num_classes = num_classes

    def config_to_distribution(self, config: TargetShiftConfig):
        probs = torch.zeros(self.num_classes)
        total_specified_probs = sum(config.values())
        num_unspecified_entries = self.num_classes - len(config)
        assert total_specified_probs <= 1.0001, f"{total_specified_probs} exceeded 1.0"
        for i in range(self.num_classes):
            if i in config:
                probs[i] = config[i]
            else:
                unspecified_prob = (1.0 - total_specified_probs) / num_unspecified_entries
                probs[i] = unspecified_prob
        return probs

    def shuffle_dataset(self, data, targets):
        permut = np.random.permutation(len(targets))
        return data[permut], targets[permut]

    def get_probs(self, configs: List[TargetShiftConfig], normalize=True):
        """Returns tensor of size (num_splits, num_classes)
        """
        probs = torch.stack(list(map(self.config_to_distribution, configs)))
        if normalize:
            return self.normalize_probs(probs)
        else:
            return probs

    def normalize_probs(self, probs):
        normalizer = torch.max(torch.sum(probs, 0))
        return probs / normalizer

    def split_class_data(self, class_id, class_probs, data, targets):
        idx = targets == class_id
        data, targets = data[idx], targets[idx]
        total_size = data.shape[0]
        split_sizes = torch.floor(class_probs * total_size)
        #split_sizes = torch.max(split_sizes, torch.tensor(1.0)) # ensure at least one elements
        split_sizes = split_sizes.int().tolist() 
        remaining_size = total_size - sum(split_sizes)
        split_sizes.append(remaining_size)
        return torch.split(data, split_sizes)[:-1], torch.split(targets, split_sizes)[:-1]

    def make_uniform(self, data, targets):
        min_class_count = min(get_targets_counts(targets).values())
        all_idx = []
        for i in range(self.num_classes):
            # Target indices have value True
            is_class = targets == i

            # Get all indices that are True
            class_idx = is_class.nonzero()

            # Only pick the minimum fraction
            all_idx.append(class_idx[:min_class_count])
        idx = torch.cat(all_idx)
        return data[idx], targets[idx]

    def split_dataset(self, 
        data, 
        targets, 
        configs: List[TargetShiftConfig], 
        shuffle=False, 
        transform=None,
        imagedata=True,
    ) -> List[InMemoryDataset]:

        num_splits = len(configs)
        if shuffle:
            data, targets = self.shuffle_dataset(data, targets)
        data, targets = self.make_uniform(data, targets)
        probs = self.get_probs(configs)
        
        class_data_splits = {}
        class_targets_splits = {}

        for i in range(self.num_classes):
            class_probs = probs[:, i]
            class_data_splits[i], class_targets_splits[i] = self.split_class_data(i, class_probs, data, targets)

        dataset_splits = []
        for j in range(num_splits):
            split_data = torch.cat([class_data_splits[class_id][j] for class_id in range(self.num_classes)])
            split_targets = torch.cat([class_targets_splits[class_id][j] for class_id in range(self.num_classes)])
            dataset_splits.append(InMemoryDataset(split_data, split_targets, transform=transform, imagedata=imagedata))
        
        return dataset_splits

    def get_ratios(self, test_configs, train_configs, combine_testsets=False):
        test_probs = self.get_probs(test_configs)
        train_probs = self.get_probs(train_configs)

        if combine_testsets:
            test_probs = test_probs.sum(0)
            test_probs = test_probs/test_probs.sum()
            test_probs = test_probs.repeat(train_probs.shape[0], 1)

        test_probs = test_probs.t()/torch.sum(test_probs, 1)
        train_probs = train_probs.t()/torch.sum(train_probs, 1)
        true_ratio = test_probs/train_probs
        return true_ratio.t()

def get_targets_counts(targets):
    uniq, counts = torch.unique(targets, return_counts=True)
    return dict(zip(uniq.tolist(), counts.tolist()))
