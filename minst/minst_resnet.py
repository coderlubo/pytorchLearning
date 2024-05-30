import torch
import torch.utils
import torchvision

from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from utils import PrintModel


BATCH_SIZE = 64

trans = transforms.Compose([
    transforms.ToTensor(),
])

train_set = datasets.MNIST("./data", train=True, transform=trans, download=True)
test_set = datasets.MNIST("./data", train=False, transform=trans, download=True)


train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)



