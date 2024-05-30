from ast import mod
from torch import nn
from utils.parameters import DEVICE

loss = nn.CrossEntropyLoss()
loss = loss.to(DEVICE)

def train_model(model, train_loader, optim):
    
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        optim.zero_grad()

        output = model(data)

        result_loss = loss(output, target)
        result_loss.backward()
        optim.step()

        if batch_index % 300 == 0:
            
            print("batch index: {} \t  Loss:{:.6f}".format(batch_index, result_loss.item()))
