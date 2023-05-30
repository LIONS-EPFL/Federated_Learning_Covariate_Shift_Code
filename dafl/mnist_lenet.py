
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class D3RE_LeNet(nn.Module):

    def __init__(self, rep_dim=1, force_pos=False):
        super().__init__()

        self.force_pos= force_pos
        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 16, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(32 * 7 * 7, 128, bias=False)
        self.bn1d1 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(128, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = x.view(int(x.size(0)), -1)
        x = F.leaky_relu(self.bn1d1(self.fc1(x)))
        x = self.fc2(x)
        if self.force_pos:
            x = self.relu(x)
        return x


class LeNet(nn.Module):
    """43,661 parameters
    """
    def __init__(self, rep_dim=1, force_pos=False):
        super(LeNet,self).__init__()
        self.force_pos= force_pos
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1,6,5),
            nn.Tanh(),                      
            nn.AvgPool2d(2,stride=2),
            nn.Conv2d(6,16,5),
            nn.Tanh(),
            nn.AvgPool2d(2,stride=2)
            )
        
        self.fc_model = nn.Sequential(
            nn.Linear(256,120),
            nn.Tanh(),
            nn.Linear(120,84),
            nn.Tanh(),
            nn.Linear(84,rep_dim)
            )
            
    def forward(self,x):     
        x = self.cnn_model(x)       
        x = x.view(x.size(0),-1)
        x = self.fc_model(x)
        if self.force_pos:
            x = self.relu(x)
        return x


class ConvNet_3Channel_large(nn.Module):
  """427,571 parameters
  """
  def __init__(self, rep_dim=1, force_pos=False):
    super(ConvNet_3Channel_large, self).__init__()
    self.conv1 = nn.Conv2d(3, 20, 5, 1)
    self.conv2 = nn.Conv2d(20, 50, 5, 1)
    self.fc1 = nn.Linear(4 * 4 * 50, 500)
    self.fc2 = nn.Linear(500, rep_dim)
    self.force_pos = force_pos

  def forward(self, x):
    x = F.tanh(self.conv1(x))
    x = F.avg_pool2d(x, 2, 2)
    x = F.tanh(self.conv2(x))
    x = F.avg_pool2d(x, 2, 2)
    x = x.view(-1, 4 * 4 * 50)
    x = F.tanh(self.fc1(x))
    x = self.fc2(x)
    if self.force_pos:
        x = self.relu(x)
    return x


class ConvNet_3Channel_small(nn.Module):
    """43,961 parameters
    """
    def __init__(self, rep_dim=1):
        super(ConvNet_3Channel_small, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, rep_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
