import torch
import torch.utils
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader


trans = torchvision.transforms .ToTensor()
dataset = torchvision.datasets.CIFAR10("./data3", train=False, transform=trans, download=True)

dataloader = DataLoader(dataset=dataset, batch_size=64)


class Net(nn.Module):
    """Some Information about Net"""
    def __init__(self) :
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, input):

        output = self.conv1(input)

        return output
    


net = Net()

for data in dataloader:
    imgs, targets = data
    output = net(imgs)

    # reshape 成能够直接输出的格式
    output = torch.reshape(output, (-1, 3, 30, 30))

    print(imgs.shape)
    print(output.shape)
