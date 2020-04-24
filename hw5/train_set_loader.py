import sys, os
import torch
from dataset import FoodDataset

dataset_dir = './data/food-11/'
output_path = './data/train_set.npy'

# 給予 data 的路徑，回傳每一張圖片的「路徑」和「class」
def get_paths_labels(path):
    imgnames = os.listdir(path)
    imgnames.sort()
    imgpaths = []
    labels = []
    for name in imgnames:
        imgpaths.append(os.path.join(path, name))
        labels.append(int(name.split('_')[0]))
    return imgpaths, labels

train_paths, train_labels = get_paths_labels(os.path.join(dataset_dir, 'training'))
train_set = FoodDataset(train_paths, train_labels, mode='eval')

torch.save(train_set, output_path)