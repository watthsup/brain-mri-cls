from torch.utils.data import Dataset
import cv2
import numpy as np
from glob import glob
import torch

class BrainDataset(Dataset):
    def __init__(
            self,
            root_dir = '/data/users/6370327221/dataset/MRI-Brain-tumor-cls/',
            is_train = True,
            transform = None
            ):
        if is_train:
            self.im_paths = sorted(glob(root_dir + 'Training/**' + '/*.jpg'))
        elif not is_train:
            self.im_paths = sorted(glob(root_dir + 'Testing/**' + '/*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, index):
        im_path = self.im_paths[index]
        image = cv2.imread(im_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image / 255.0

        label = im_path.split('/')[-2].split('_')[0]
        label = torch.tensor(int(label)).long()

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
            image = image.float()

        return image, label