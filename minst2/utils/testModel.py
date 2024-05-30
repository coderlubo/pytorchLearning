import torch
from utils.settings import DEVICE
from torch import nn


loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(DEVICE)

def test_model(model, test_load, logger):

    model.eval()

    correct = 0.0
    loss = 0.0

    with torch.no_grad():
        for data, target in test_load:

            data, target = data.to(DEVICE), target.to(DEVICE)

            output = model(data)

            loss += loss_fn(output, target).item()

            max_output = output.argmax(dim=1)

            correct += max_output.eq(target.view_as(max_output)).sum().item()

        test_loss = loss / len(test_load.dataset)
        accuracy = correct / len(test_load.dataset) * 100.0

        logger.info("Loss: {:.4f} \t Accuracy: {:.4f}".format(test_loss, accuracy))

    return test_loss, accuracy

