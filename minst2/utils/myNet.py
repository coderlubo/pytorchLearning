from torchvision import models
from torch import nn


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=3), # 16 * 32 * 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 16 * 16 * 16
            nn.Conv2d(in_channels=16, out_channels=128, kernel_size=3, padding=1), # 128 * 16 * 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 128 * 8 * 8
            nn.Conv2d(in_channels=128, out_channels=1024, kernel_size=3, padding=1), # 1024 * 8 * 8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 1024 * 4 * 4
            nn.Flatten(),
            nn.Linear(in_features=1024*4*4, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, input):

        output = self.model(input)

        return output