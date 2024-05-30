import torch


BATCH_SIZE = 64
EPOCHS = 20
DEVICE = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
LEARN_RATE = 0.01