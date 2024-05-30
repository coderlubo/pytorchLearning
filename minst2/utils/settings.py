import time
import torch


BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 10
LOGGER_PATH = "./output/" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '.log'