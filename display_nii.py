# author: detoX
import glob

import numpy as np
import torch
import nibabel as nib
from PIL import Image

import nii_dataset


def main():
    train_path = glob.glob("D:\\xuexi\\post-graduate\\py_projects\\ResNet-PET\\datasets\\Brain-PET\\Train\\*\\*")
    test_path = glob.glob("D:\\xuexi\\post-graduate\\py_projects\\ResNet-PET\\datasets\\Brain-PET\\Test\\*")

    count = 0
    for path in train_path:
        img = nib.load(path)
        img = img.dataobj[:, :, :, 0]
        idx = np.random.choice(range(img.shape[-1]), 50)
        img = img[:, :, idx]
        print(img.shape)
        slice_img = img[:, :, 3]
        # 由于原数据并不是图片，需要将图片归一化到[0, 1]区间，然后再放大到[0, 255]区间，因为灰度图片的亮度区间是0-255
        slice_img = (slice_img / slice_img.max() * 255)
        slice_img = Image.fromarray(slice_img)
        if img.shape[0] != 128:
            Image._show(slice_img)
        else:
            if count < 10:
                Image._show(slice_img)
                count += 1
    # idx = np.random.choice(range(img.shape[-1]), 50)
    # # idx.sort()
    # img = img[:, :, idx]
    # img = img.astype(np.float32)


def display_single_nii():
    train_path = glob.glob("D:\\xuexi\\post-graduate\\py_projects\\ResNet-PET\\datasets\\Brain-PET\\Train\\*\\*")
    test_path = glob.glob("D:\\xuexi\\post-graduate\\py_projects\\ResNet-PET\\datasets\\Brain-PET\\Test\\*")
    for path in train_path:
        img = nib.load(path)
        img = img.dataobj[:, :, :, 0]
        if img.shape[2] == 47:
            # idx = np.random.choice(range(img.shape[-1]), 50)
            # img = img[:, :, idx]
            for s in range(47):
                slice_img = img[:, :, s]
                slice_img = (slice_img / slice_img.max() * 255).astype('uint8')
                slice_img = Image.fromarray(slice_img)
                # slice_img.show()
                slice_img.save("./slice_imgs/nii_50/slice_{}.png".format(s))
            break
    # for i in range(50):
    #     slice_img = img[:, :, i]
    #     slice_img = (slice_img / slice_img.max() * 255)
    #     slice_img = Image.fromarray(slice_img)
    #     slice_img.show()


if __name__ == '__main__':
    # main()
    display_single_nii()
