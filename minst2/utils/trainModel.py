import torch
from utils.settings import DEVICE
from torch import nn

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(DEVICE)

def train_model(model, train_load, optim, logger):

    model.train()

    correct = 0.0
    loss = 0.0

    for batch_index, (data, target) in enumerate(train_load):

        data, target = data.to(DEVICE), target.to(DEVICE)

        optim.zero_grad()

        output = model(data)
        loss = loss_fn(output, target)

        loss.backward()
        optim.step()

        if batch_index % 300 == 0:
            logger.info("batch_index: {:<3d} \tloss: {:.4f}".format(batch_index, loss))
