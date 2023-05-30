"""
Partially borrowed from https://github.com/Chavdarova/LAGAN-Lookahead_Minimax
"""
import torch
import torch.nn as nn
from collections import OrderedDict
import os

import numpy as np
from scipy import linalg
from torch.utils.data import DataLoader


class LabelBasedRatio(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(num_classes, 1)

    def forward(self, y):
        y = torch.nn.functional.one_hot(y, self.num_classes).float()
        return self.linear(y)


class LinearOnFeatures(torch.nn.Module):
    def __init__(self, feature_extractor, num_features):
        super().__init__()

        self.feature_extractor = feature_extractor
        for name, param in self.feature_extractor.named_parameters():
            param.requires_grad = False
        self.linear = torch.nn.Linear(num_features, 1)

    def forward(self, input):
        return self.linear(self.feature_extractor(input))


class MLP_mnist(nn.Module):
  def __init__(self, input_dims, n_hiddens, n_class):
    super(MLP_mnist, self).__init__()
    assert isinstance(input_dims, int), 'Expected int for input_dims'
    self.input_dims = input_dims
    current_dims = input_dims
    layers = OrderedDict()

    if isinstance(n_hiddens, int):
      n_hiddens = [n_hiddens]
    else:
      n_hiddens = list(n_hiddens)
    for i, n_hidden in enumerate(n_hiddens):
      layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
      layers['relu{}'.format(i+1)] = nn.ReLU()
      layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
      current_dims = n_hidden
    layers['out'] = nn.Linear(current_dims, n_class)
    self.layers = layers
    self.model= nn.Sequential(layers)
    #print(self.model)

  def forward(self, input):
    input = input.view(input.size(0), -1)
    assert input.size(1) == self.input_dims
    return self.model.forward(input)

  def get_logits_and_fc2_outputs(self, x):
    x = x.view(x.size(0), -1)
    assert x.size(1) == self.input_dims
    fc2_out = None
    for l in self.model:
      x = l(x)
      if l == self.layers["fc2"]:
        fc2_out = x
    return x, fc2_out


def pretrained_mnist_model(input_dims=784, n_hiddens=[256, 256], n_class=10, 
                           pretrained=None):
    model = MLP_mnist(input_dims, n_hiddens, n_class)
    if pretrained is not None:
        if os.path.exists(pretrained):
            print('Loading trained model from %s' % pretrained)
            state_dict = torch.load(pretrained,
                    map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
            if 'parallel' in pretrained:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                state_dict = new_state_dict
        else:
            raise FileNotFoundError(f"Could not find pretrained model: {pretrained}.")
        model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model = model.cuda()
    return model
