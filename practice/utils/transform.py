from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

img_path = "/home/lubo/pytorchLearning/practice/data/catvsdog/train/cat.6.jpg"
img_PIL = Image.open(img_path)


writer = SummaryWriter('logs')

# ToTensor
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img_PIL)

writer.add_image("tensor_img", tensor_img)

# Normalize
norm_trans = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = norm_trans(tensor_img)
writer.add_image("norm_img", img_norm)

# Resize
resize_trans = transforms.Resize((512,  512))
img_resize = resize_trans(img_PIL)
img_resize = tensor_trans(img_resize)
writer.add_image("resize", img_resize)

# Compose - Resize
resize_trans_2 = transforms.Resize(512)
compose_trans = transforms.Compose([resize_trans_2, tensor_trans])
img_resize_2 = compose_trans(img_PIL)

writer.add_image("resize", img_resize_2, 1)

# RandomCrop
random_trans = transforms.RandomCrop(512)



writer.close()