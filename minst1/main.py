import time
import torch
from utils.printData import printData
from utils.myNet import Net
from utils.parameters import DEVICE, EPOCHS, LEARN_RATE
from utils.getData import get_data
from utils.printM import printM
from utils.testModel import test_model
from utils.trainModel import train_model

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./logs")

train_load, test_load = get_data()

model = Net().to(DEVICE)
optim = torch.optim.Adam(model.parameters())
# optim = torch.optim.SGD(model.parameters(), lr=LEARN_RATE)
# optim = torch.optim.ASGD(model.parameters())

for epoch in range(1, EPOCHS+1):
    print("EPOCH: {}".format(epoch))
    begin_time = time.time()
    train_model(model, train_load, optim)
    end_time = time.time()
    print("TRAIN TIME: {}".format(end_time-begin_time))

    # 训练时间可视化
    writer.add_scalar('time', end_time-begin_time, epoch)
    
    test_loss, accuracy = test_model(model, test_load)

    # 测试数据可视化
    writer.add_scalar('Average loss', test_loss, epoch)
    writer.add_scalar('Accuracy', accuracy, epoch)
    print("<<" + "+" * 60 + ">>")

writer.close()