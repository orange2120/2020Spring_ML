# preprocess
# Prepare training data
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def preprocess(image_list):
    """ Normalize Image and Permute (N,H,W,C) to (N,C,H,W)
    Args:
      image_list: List of images (9000, 32, 32, 3)
    Returns:
      image_list: List of images (9000, 3, 32, 32)
    """
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    image_list = (image_list / 255.0) * 2 - 1
    image_list = image_list.astype(np.float32)
    return image_list

# class Image_Dataset(Dataset):
#     def __init__(self, image_list):
#         self.image_list = image_list
#     def __len__(self):
#         return len(self.image_list)
#     def __getitem__(self, idx):
#         images = self.image_list[idx]
#         return images

class Image_Dataset(Dataset):
    def __init__(self, image_list, y=None, transform=None):
        self.image_list = image_list
        if y is not None:
          self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.image_list)
        
    def __getitem__(self, idx):
        images = self.image_list[idx]
        if self.transform is not None:
          images = self.transform(images)
        # if self.y is not None:
        #   Y = self.y[idx]
        #   return images, Y
        return images