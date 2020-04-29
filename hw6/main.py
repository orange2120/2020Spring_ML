import os, sys
# 讀取 label.csv
import pandas as pd
# 讀取圖片
# from PIL import Image
import numpy as np

# import torch
# # Loss function
# import torch.nn.functional as F
# # 讀取資料
# import torchvision.datasets as datasets
# from torch.utils.data import Dataset, DataLoader
# # 載入預訓練的模型
# import torchvision.models as models
# # 將資料轉換成符合預訓練模型的形式
# import torchvision.transforms as transforms
# 顯示圖片
import matplotlib.pyplot as plt

from utils import *

data_path = './data'
output_dir = './output'

if len(sys.argv) == 3:
    data_path = sys.argv[1]
    output_dir = sys.argv[2]

if __name__ == '__main__':
    # 讀入圖片相對應的 label
    df = pd.read_csv(os.path.join(data_path, 'labels.csv'))
    df = df.loc[:, 'TrueLabel'].to_numpy()
    label_name = pd.read_csv(os.path.join(data_path, 'categories.csv'))
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()
    # new 一個 Attacker class
    attacker = Attacker(os.path.join(data_path, 'images'), df)
    # 要嘗試的 epsilon
    epsilons = [0.1, 0.01]

    accuracies, examples = [], []

    # 進行攻擊 並存起正確率和攻擊成功的圖片
    for eps in epsilons:
        ex, acc = attacker.attack(eps)
        accuracies.append(acc)
        examples.append(ex)

    cnt = 0
    plt.figure(figsize=(30, 30))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]) * 2,cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,orig_img, ex = examples[i][j]
            # plt.title("{} -> {}".format(orig, adv))
            plt.title("original: {}".format(label_name[orig].split(',')[0]))
            orig_img = np.transpose(orig_img, (1, 2, 0))
            plt.imshow(orig_img)
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]) * 2,cnt)
            plt.title("adversarial: {}".format(label_name[adv].split(',')[0]))
            ex = np.transpose(ex, (1, 2, 0))
            plt.imshow(ex)
    plt.tight_layout()
    plt.show()