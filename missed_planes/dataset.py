import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PlanesDataset(Dataset):
    def __init__(self, data, path, is_test=False, augmentation=None):
        super().__init__()
        self.data = data
        self.path = path
        self.is_test = is_test
        self.augmentation = augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        try:
            row = self.data.iloc[item]
            image = cv2.imread(os.path.join(self.path, row["filename"]) + ".png")
            if self.augmentation:
                image = self.augmentation(image=image)["image"]

            if self.is_test:
                return torch.tensor(np.moveaxis(image, -1, 0), dtype=torch.float)

            return torch.tensor(np.moveaxis(image, -1, 0), dtype=torch.float), torch.tensor(
                row["sign"], dtype=torch.float
            )
        except:
            print(row["filename"])
