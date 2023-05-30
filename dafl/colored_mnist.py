"""Modified version of https://colab.research.google.com/github/reiinakano/invariant-risk-minimization/blob/master/invariant_risk_minimization_colored_mnist.ipynb#scrollTo=knP-xNzavgAb
"""
import os

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils


def color_grayscale_arr(arr, red=True):
  """Converts grayscale image to either red or green"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  if red:
    arr = np.concatenate([arr,
                          np.zeros((h, w, 2), dtype=dtype)], axis=2)
  else:
    arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                          arr,
                          np.zeros((h, w, 1), dtype=dtype)], axis=2)
  return arr


class ColoredMNISTTrueRatioModel(torch.nn.Module):
    def __init__(self, train_flip_prob=0.1, test_flip_prob=0.9):
      super().__init__()
      self.train_flip_prob = train_flip_prob
      self.test_flip_prob = test_flip_prob
      self.use_target = True

    def forward(self, img, target):
      _, colors = img.sum(-1).sum(-1).nonzero(as_tuple=True)
      red = 1
      not_flipped_ratio = (1-self.test_flip_prob)/(1-self.train_flip_prob)
      flipped_ratio = self.test_flip_prob/self.train_flip_prob # unlikely in train but likely in test, so weight a lot
      return torch.where(colors == target, not_flipped_ratio, flipped_ratio)


class ColoredMNIST(datasets.VisionDataset):
  """
  Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test1', 'test2', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """
  def __init__(self, root='./data', env='train1', transform=None, target_transform=None, grayscale=False, divide=40):
    super(ColoredMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)
    self.grayscale = grayscale

    self.prepare_colored_mnist(divide=divide)
    if env in ['train1', 'train2', 'test1', 'test2']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
    elif env == 'all_train':
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST', 'train2.pt'))
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test1, test2, and all_train')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    # Make grayscale (copy 1's to first channel)
    if self.grayscale:
        num_channels = img.shape[0]
        img = img.sum(axis=0, keepdim=True).repeat(num_channels, 1, 1)
        img[1] = 0
        img[2] = 0

    return img, target

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_colored_mnist(self, divide=40):
    colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
    if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train2.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'test1.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'test2.pt')):
      print('Colored MNIST dataset already exists')
      return

    print('Preparing Colored MNIST')
    train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

    # First environment is heavily reduced
    train1_last_idx = 20000//divide
    train2_first_idx = 20000
    train2_last_idx = 40000
    test1_last_idx = 50000
    test2_last_idx = 60000
    
    train1_set = []
    train2_set = []
    test1_set = []
    test2_set = []
    for idx, (im, label) in enumerate(train_mnist):
      if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(train_mnist)}')
      im_array = np.array(im)

      # Assign a binary label y to the image based on the digit
      binary_label = 0 if label < 5 else 1

      # Flip label with 25% probability
      if np.random.uniform() < 0.25:
        binary_label = binary_label ^ 1
      ori_binary_label = binary_label

      # Color the image either red or green according to its possibly flipped label
      color_red = binary_label == 0

      # Flip the color with a probability e that depends on the environment
      if idx < train1_last_idx:
        # 50% in the train1 environment
        if np.random.uniform() < 0.5:
          color_red = not color_red
      elif idx > train1_last_idx and idx <= train2_first_idx:
        continue
      elif idx < train2_last_idx:
        # 1% in the train2 environment
        if np.random.uniform() < 0.2:
          color_red = not color_red
      elif idx < test1_last_idx:
        # 1% in the test1 environment
        if np.random.uniform() < 0.2:
          color_red = not color_red
      else:
        # 99% in the test2 environment
        if np.random.uniform() < 0.8:
          color_red = not color_red

      colored_arr = color_grayscale_arr(im_array, red=color_red)

      # Flip label with 25% probability (AFTER coloring!)
      # ori_binary_label = binary_label
      # if np.random.uniform() < 0.25:
      #   binary_label = binary_label ^ 1

      if idx < train1_last_idx:
        train1_set.append((Image.fromarray(colored_arr), binary_label))
      elif idx < train2_last_idx:
        # if not flipped only include 0.1/0.9 of the time (to emulate uniform distribution)
        # if color_red == binary_label and np.random.uniform() < 0.11111:
        train2_set.append((Image.fromarray(colored_arr), binary_label))
        # simulate uniform reweighting (0.8/0.2)
        # if color_red != ori_binary_label:
        #     train2_set.append((Image.fromarray(colored_arr), binary_label))
        #     train2_set.append((Image.fromarray(colored_arr), binary_label))
        #     train2_set.append((Image.fromarray(colored_arr), binary_label))
      elif idx < test1_last_idx:
        test1_set.append((Image.fromarray(colored_arr), binary_label))
      else:
        test2_set.append((Image.fromarray(colored_arr), binary_label))


      # Debug
      # print('original label', type(label), label)
      # print('binary label', binary_label)
      # print('assigned color', 'red' if color_red else 'green')
      # plt.imshow(colored_arr)
      # plt.show()
      # break

    os.makedirs(colored_mnist_dir)
    torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
    torch.save(train2_set, os.path.join(colored_mnist_dir, 'train2.pt'))
    torch.save(test1_set, os.path.join(colored_mnist_dir, 'test1.pt'))
    torch.save(test2_set, os.path.join(colored_mnist_dir, 'test2.pt'))

