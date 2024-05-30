# 对应保存方式一， 加载方式一
import torch
import torchvision

model = torch.load("./vgg16_method1")
print(model)

# 对应保存方式二， 加载方式二
vgg16_f = torchvision.models.vgg16(pretrained=False)
dic = torch.load("./vgg16_method2.pth")

vgg16_f.load_state_dict(dic)
print(model)