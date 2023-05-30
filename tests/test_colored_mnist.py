#%%
import torch
from torchvision import transforms
from dafl.colored_mnist import ColoredMNIST


transform_train = transforms.Compose([
    transforms.ToTensor(),
])
target_transform = lambda target: float(target)

cmnist = ColoredMNIST(root='../data', env='train1', transform=transform_train, target_transform=target_transform, grayscale=False)

loader = torch.utils.data.DataLoader(cmnist, batch_size=64, shuffle=True)
img,target = next(iter(loader))
_, colors = img.sum(-1).sum(-1).nonzero(as_tuple=True)
red = 1
not_flipped_ratio = 0.9/0.1
flipped_ratio = 0.1/0.9
torch.where(colors == target, not_flipped_ratio, flipped_ratio)

# %%
