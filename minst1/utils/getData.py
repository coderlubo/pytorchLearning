from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.parameters import BATCH_SIZE


trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_data():
    # train_set = datasets.MNIST("./data", train=True, transform=trans, download=True)
    # test_set = datasets.MNIST("./data", train=False, transform=trans, download=True)

    # train_load = DataLoader(train_set, BATCH_SIZE, shuffle=True)
    # test_load = DataLoader(test_set, BATCH_SIZE, shuffle=False)

    train_set = datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
    test_set = datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)

    train_load = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_load = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_load, test_load