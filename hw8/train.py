import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.sampler as sampler
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
from utils import *
from dataset import *
from test import *
from config import configurations

import numpy as np
import sys
import os

device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")  # 判斷是用 CPU 還是 GPU 執行運算


# def adjust_learning_rate(optimizer, lr):
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def train(model, optimizer, train_iter, loss_function, total_steps, summary_steps, train_dataset):

  # if epoch < 15:
  #   lr = 0.001
  # elif epoch < 30:
  #   lr = 5E-4
  # else:
  #   lr = 2E-4

  # adjust_learning_rate(optimizer, lr)
  model.train()
  model.zero_grad()
  losses = []
  loss_sum = 0.0
  for step in range(summary_steps):
    sources, targets = next(train_iter)
    sources, targets = sources.to(device), targets.to(device)
    outputs, preds = model(sources, targets, schedule_sampling())
    # targets 的第一個 token 是 <BOS> 所以忽略
    outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
    targets = targets[:, 1:].reshape(-1)
    loss = loss_function(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()

    loss_sum += loss.item()
    if (step + 1) % 5 == 0:
      loss_sum = loss_sum / 5
      print ("\r", "train [{}] loss: {:.3f}, Perplexity: {:.3f}      ".format(total_steps + step + 1, loss_sum, np.exp(loss_sum)), end=" ")
      losses.append(loss_sum)
      loss_sum = 0.0

  return model, optimizer, losses

def train_process(config):
  # 準備訓練資料
  train_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'training')
  train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
  train_iter = infinite_iter(train_loader)
  # 準備檢驗資料
  val_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'validation')
  val_loader = data.DataLoader(val_dataset, batch_size=1)
  # 建構模型
  model, optimizer = build_model(config, train_dataset.en_vocab_size, train_dataset.cn_vocab_size)
  loss_function = nn.CrossEntropyLoss(ignore_index=0)

  train_losses, val_losses, bleu_scores = [], [], []
  total_steps = 0

  timestr = time.strftime("%Y%m%d-%H-%M-%S")

  while (total_steps < config.num_steps):
    # 訓練模型
    model, optimizer, loss = train(model, optimizer, train_iter, loss_function, total_steps, config.summary_steps, train_dataset)
    train_losses += loss
    # 檢驗模型
    val_loss, bleu_score, result = test(model, val_loader, loss_function)
    val_losses.append(val_loss)
    bleu_scores.append(bleu_score)

    total_steps += config.summary_steps
    print ("\r", "val [{}] loss: {:.3f}, Perplexity: {:.3f}, blue score: {:.3f}       ".format(total_steps, val_loss, np.exp(val_loss), bleu_score))
    
    # 儲存模型和結果
    if total_steps % config.store_steps == 0 or total_steps >= config.num_steps:
      save_model_t(model, optimizer, config.store_model_path, total_steps, timestr)
    #   save_model(model, optimizer, config.store_model_path, total_steps)
      with open(f'{config.store_model_path}/output_{timestr}_{total_steps}.txt', 'w') as f:
        for line in result:
          print (line, file=f)
    
  return train_losses, val_losses, bleu_scores


if __name__ == '__main__':
    config = configurations()
    print ('config:\n', vars(config))

    train_losses, val_losses, bleu_scores = train_process(config)

    timestr = time.strftime("%Y%m%d-%H-%M-%S")

    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('iters')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.savefig(f'./{config.figure_output_path}/train_loss_{timestr}.png')
    plt.clf()

    plt.figure()
    plt.plot(val_losses)
    plt.xlabel('iters')
    plt.ylabel('loss')
    plt.title('validation loss')
    plt.savefig(f'./{config.figure_output_path}/val_loss_{timestr}.png')
