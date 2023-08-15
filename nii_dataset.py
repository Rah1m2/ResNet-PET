# author: detoX
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

DATA_CACHE = {}

class NiiDataset(Dataset):
    classes = [
        "0 - NC",
        "1 - MCI",
    ]

    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        if self.img_path[index] in DATA_CACHE:
            img = DATA_CACHE[self.img_path[index]]
        else:
            img = nib.load(self.img_path[index])
            img = img.dataobj[:, :, :, 0]
            DATA_CACHE[self.img_path[index]] = img

        idx = np.random.choice(range(img.shape[-1]), 50)
        # idx.sort()
        img = img[:, :, idx]
        img = img.astype(np.float32)

        # img /= 255.0
        # img -= 1

        # print(img.shape)
        if self.transform is not None:
            img = self.transform(image=img)['image']  # 用A进行transform之后返回一个字典，有两个属性，其中一个image属性就是图片

        img = img.transpose([2, 0, 1])
        # 如果这个nii文件是正常人的，label就标记为True，然后再将bool转化为int，也就是1
        return img, torch.from_numpy(np.array(int('NC' in self.img_path[index])))

    def __len__(self):
        return len(self.img_path)
