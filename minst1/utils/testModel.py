import torch

from torch import nn
from utils.printData import printData
from utils.parameters import DEVICE

loss = nn.CrossEntropyLoss()
loss = loss.to(DEVICE)

def test_model(model, test_loader):

    correct = 0.0
    test_loss = 0.0
    
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            output = model(data)

            test_loss += loss(output, target).item()

            max_index = output.argmax(dim=1)

            correct += max_index.eq(target.view_as(max_index)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)
 
        print("Test -- Average loss: {:.4f}, Accuracy : {:.3f}\n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)
        ))

    return test_loss, accuracy 