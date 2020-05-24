# problem 2
import os, sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from preprocess import *
from model import AE

checkpoint_path = './data/checkpoints'

trainX = np.load('./data/trainX_new.npy')
trainX_preprocessed = preprocess(trainX)

model_path = f'./{checkpoint_path}/last_checkpoint.pth'
model_prefix = ''
if len(sys.argv) == 2:
    model_path = sys.argv[1]
    model_prefix = os.path.splitext(os.path.basename(model_path))[0]

# load model
model = AE().cuda()
model.load_state_dict(torch.load(model_path))

# 畫出原圖
plt.figure(figsize=(10,4))
indexes = [1,2,3,6,7,9]
imgs = trainX[indexes,]
for i, img in enumerate(imgs):
    plt.subplot(2, 6, i+1, xticks=[], yticks=[])
    plt.imshow(img)

# 畫出 reconstruct 的圖
inp = torch.Tensor(trainX_preprocessed[indexes,]).cuda()
latents, recs = model(inp)
recs = ((recs+1)/2 ).cpu().detach().numpy()
recs = recs.transpose(0, 2, 3, 1)
for i, img in enumerate(recs):
    plt.subplot(2, 6, 6+i+1, xticks=[], yticks=[])
    plt.imshow(img)
  
plt.tight_layout()
plt.savefig(f'./data/figure/reconstruct_{model_prefix}.png')