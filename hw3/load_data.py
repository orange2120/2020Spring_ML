import sys, os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

workspace_dir = '../data/food-11'
output_dir = '../data/food-11/data.npz'

if len(sys.argv) == 2:
  workspace_dir = sys.argv[1]
  output_dir = sys.argv[1] + '/data.npz'

def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x

#分別將 training set、validation set、testing set 用 readfile 函式讀進來
print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))
print('File loaded.')

np.savez(output_dir, tr_x=train_x, tr_y=train_y, val_x=val_x, val_y=val_y, te_x=test_x)
