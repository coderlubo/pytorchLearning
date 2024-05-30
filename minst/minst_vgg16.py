
# 1.加载库
from ast import mod
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# 2.定义超参数
BATCH_SIZE = 16 # 每批处理的数据
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # GPU / CPU
EPOCHS = 10 # 训练数据集的轮次

# 3.构建pipeline， 对图像做处理
pipeline = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换成tensor格式
    transforms.Normalize((0.1307,), (0.3081,)) # 图片正则化，降低模型复杂度
])

# 4.下载、加载数据集
from torch.utils.data import DataLoader

# 下载数据集
train_set = datasets.MNIST("data", train=True, download=True, transform=pipeline)   # 数据集

test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline)    # 测试集


# 加载数据
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) 
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True) 



# 5.构建网络模型
model = torchvision.models.vgg16(pretrained=False)
model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
model.classifier.add_7 = nn.Linear(1000, 10)
model.classifier.add_8 = nn.LogSoftmax(dim=1)
print(model)
model = model.cuda()

# class Digit(nn.Module):

#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.conv1 = nn.Conv2d(1, 10 ,5)    # 1: 灰度图片的通道，10：输出通道，5：kernel Size
#         self.conv2 = nn.Conv2d(10, 20, 3)   # 10: 输入通道，20：输出通道， 3：Kernel Size
#         self.fc1 = nn.Linear(20*10*10, 500) # 20*10*10：输入通道，500：输出通道
#         self.fc2 = nn.Linear(500, 10)       # 500：输入通道，10：输出通道
    
#     def forward(self, x):
#         input_size = x.size(0) # batch_size
#         x = self.conv1(x) # 输入：batch_size*1*28*28，输出：batch_size*10*24*24 (28-5+1=24)
#         x = F.relu(x) # 激活函数，保持shape不变，输出：batch_size*10*24*24
#         x = F.max_pool2d(x, 2, 2)   # 池化层：输入：batch_size*10*24*24， 输出：batch_size*10*12*12

#         x = self.conv2(x) # 输入：batch_size*10*12*12， 输出：batch_size*20*10*10 (12-3+1=10)
#         x = F.relu(x) # 激活函数

#         x = x.view(input_size, -1) # 拉伸， -1：自动计算维度 20*10*10=2000

#         x = self.fc1(x) # 输入：batch_size*2000, 输出：batch_size*500
#         x = F.relu(x) # 激活函数，保持shape不变，

#         x = self.fc2(x) # 输入：batch_size*500, 输出：batch_size*10

#         output = F.log_softmax(x, dim=1)    # 计算分类后每个数字的概率值

#         return output



# 6.定义优化器
optimizer = optim.Adam(model.parameters())

# 7.定义训练方法
def train_model(model, device, train_loader, optimizer, epoch):
    
    # 模型训练
    model.train()

    for batch_index, (data, target) in enumerate(train_loader):
        # 部署到Device
        data, target =  data.to(device), target.to(device)
        
        # 梯度初始化为0
        optimizer.zero_grad()

        # 训练后的结果
        output = model(data)

        # 计算损失, 交叉熵损失（适用于多分类任务）
        loss = F.cross_entropy(output, target)

        # # 找到概率最大值的下标
        # pred = output.argmax(dim=1)

        # 反向传播
        loss.backward()

        # 参数优化
        optimizer.step()

        # 每3000次打印一次损失值
        if batch_index % 3000 == 0:
            print("Train Epoch:{} \t Loss: {:.6f}".format(epoch, loss.item()))


# 8.定义测试方法
def test_model(model, device, test_loader):
    
    # 模型验证
    model.eval()

    # 正确率
    correct = 0.0

    # 测试损失
    test_loss = 0.0

    with torch.no_grad():   # 不会计算梯度，不会进行反向传播
        for data, target in test_loader:
            # 部署到Device上
            data, target = data.to(device), target.to(device)

            # 测试数据
            output = model(data)

            # 计算测试损失
            test_loss += F.cross_entropy(output, target).item()

            # 找到概率最大值的索引
            pred = output.argmax(dim=1)

            # 累计正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)

        print("Test -- Average loss: {:.4f}, Accuracy : {:.3f}\n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)
        ))

# 9.调用方法
for epoch in range(1, EPOCHS + 1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_model(model, DEVICE, test_loader)


