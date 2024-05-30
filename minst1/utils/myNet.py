from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch

from utils.parameters import DEVICE

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=10*10*16, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, input):

        output = self.model(input)

        return output

if __name__ == '__main__':

    data = torch.ones((64, 1, 28, 28)).to(DEVICE)
    net = Net().to(DEVICE)

    writer = SummaryWriter("./logs")
    writer.add_graph(net, data)

    writer.close()