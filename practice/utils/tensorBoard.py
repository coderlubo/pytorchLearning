from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")

# for i in range(100):
#     writer.add_scalar('y=2x', 2*i, i)


# image
img_path = "/home/lubo/pytorchLearning/practice/data/catvsdog/train/cat.7.jpg"

img_PIL = Image.open(img_path)
img_arr = np.array(img_PIL)

writer.add_image("cat", img_arr, 2, dataformats='HWC')

writer.close()
 