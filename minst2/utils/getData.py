from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils.settings import BATCH_SIZE


trans = transforms.Compose([
    transforms.ToTensor()
])

def get_data():
    train_set = datasets.MNIST(root="./data", train=True, transform=trans, download=True)
    test_set = datasets.MNIST(root="./data", train=False, transform=trans, download=True)

    train_load = DataLoader(dataset=train_set, batch_size=BATCH_SIZE)
    test_load = DataLoader(dataset=test_set, batch_size=BATCH_SIZE)

    return train_load, test_load
