import sys, os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
from myModel import Classifier, ImgDataset

num_epoch = 150
num_all_epoch = 150 
batch_size = 84

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

dataset_path = '../data/food-11/data.npz'

if len(sys.argv) == 2:
  dataset_path = sys.argv[1] + '/data.npz'

loadfile = np.load(dataset_path)
train_x = loadfile['tr_x']
train_y = loadfile['tr_y']
val_x = loadfile['val_x']
val_y = loadfile['val_y']

#training 時做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), #隨機將圖片水平翻轉
    transforms.RandomRotation(15), #隨機旋轉圖片
    transforms.ToTensor(), #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)
])
#testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
])

train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

model = Classifier().cuda()
loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # optimizer 使用 Adam

print('Start training...')

train_loss_his = []
val_loss_his = []
train_acc_his = []
val_acc_his = []

lr = 0.001

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    if epoch < 15:
        lr = 0.001
    elif epoch < 30:
        lr = 5E-4
    else:
        lr = 2E-4
    adjust_learning_rate(optimizer, lr)

    model.train() # 確保 model 是在 train model (開啟 Dropout 等...)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].cuda()) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].cuda()) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
        optimizer.step() # 以 optimizer 用 gradient 更新參數值

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
        
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        train_acc_his.append(train_acc/train_set.__len__())
        train_loss_his.append(train_loss/train_set.__len__())
        val_acc_his.append(val_acc/val_set.__len__())
        val_loss_his.append(val_loss/val_set.__len__())

        #將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))

timestr = time.strftime("%Y%m%d-%H-%M-%S")

torch.save(model, './data/model_' + timestr + '.pkl') # save model
'''
train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)
train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

train_val_loss_his = []
train_val_acc_his = []

model_best = Classifier().cuda()
loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001) # optimizer 使用 Adam

for epoch in range(num_all_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    if epoch < 15:
        lr = 0.001
    elif epoch < 30:
        lr = 5E-4
    else:
        lr = 2E-4
    adjust_learning_rate(optimizer, lr)

    model_best.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()


    train_val_acc_his.append(train_acc/train_val_set.__len__())
    train_val_loss_his.append(train_loss/train_val_set.__len__())

        #將結果 print 出來
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
      (epoch + 1, num_epoch, time.time()-epoch_start_time, \
      train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))

timestr = time.strftime("%Y%m%d-%H-%M-%S")

torch.save(model_best, './data/model_{}.pkl'.format(timestr)) # save model
'''
# Plot
# Loss curve
plt.plot(train_loss_his)
plt.plot(val_loss_his)
# plt.plot(train_val_loss_his)
plt.title('Loss')
plt.legend(['train', 'val', 'train-val'])
plt.savefig('./figure/loss_' + timestr + '.png')
# plt.show()
plt.clf()

# Accuracy curve
plt.plot(train_acc_his)
plt.plot(val_acc_his)
# plt.plot(train_val_acc_his)
plt.title('Accuracy')
plt.legend(['train', 'val', 'train-val'])
plt.savefig('./figure/accuracy_' + timestr + '.png')
# plt.show()
plt.clf()
