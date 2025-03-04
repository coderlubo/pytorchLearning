from cProfile import label
import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class MyData(Dataset):
    def __init__(self, root_dir, label_dir) -> None:
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_paths = os.listdir(self.path)
    
    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)

        label = self.label_dir

        return img, label
    
    def __len__(self):
        return len(self.img_paths)
    
root_dir = "../data/catvsdog/train"
ants_label_dir = "ants"
ants_dataset = MyData(root_dir, ants_label_dir)