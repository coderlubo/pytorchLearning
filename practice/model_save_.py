import torch
import torchvision

vgg16_f = torchvision.models.vgg16(pretrained=False)
# vgg16_t = torchvision.models.vgg16(pretrained=True)

# 保存方式一
torch.save(vgg16_f, "vgg16_method1")

# 保存方式二 （推荐）
torch.save(vgg16_f.state_dict(), "vgg16_method2.pth")