# main_train.py
import os
import torch
import argparse
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split
import utils as u
import model as m
from preprocess import Preprocess 
from data import TwitterDataset
from train import training

path_prefix = './data/'

# 要不要固定embedding、batch大小、要訓練幾個epoch、learning rate的值、model的資料夾路徑
fix_embedding = True # fix embedding during training
batch_size = 256
epoch = 20
lr = 0.001

print("loading data ...")
X_train = torch.load(os.path.join(path_prefix, 'X_train.pt'))
X_val = torch.load(os.path.join(path_prefix, 'X_val.pt'))
y_train = torch.load(os.path.join(path_prefix, 'y_train.pt'))
y_val = torch.load(os.path.join(path_prefix, 'y_val.pt'))
embedding = torch.load(os.path.join(path_prefix, 'embedding.pt'))

# 通過torch.cuda.is_available()的回傳值進行判斷是否有使用GPU的環境，如果有的話device就設為"cuda"，沒有的話就設為"cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = os.path.join(path_prefix, 'model/') # model directory for checkpoint model

# 製作一個model的對象
# model = m.LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
model = m.LSTM_Net(embedding, embedding_dim=250, hidden_dim=200, num_layers=4, dropout=0.5, fix_embedding=fix_embedding)
model = model.to(device) # device為"cuda"，model使用GPU來訓練(餵進去的inputs也需要是cuda tensor)

# 把data做成dataset供dataloader取用
train_dataset = TwitterDataset(X=X_train, y=y_train)
val_dataset = TwitterDataset(X=X_val, y=y_val)

# 把data 轉成 batch of tensors
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 8)

val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)

print('start training...')
# 開始訓練
training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)

print('done.')
