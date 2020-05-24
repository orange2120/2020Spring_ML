import os, sys
import torch
import numpy as np
from model import AE
from utils import *

batch_size = 32

val_x = np.load('./data/valX.npy')
val_y = np.load('./data/valY.npy')

model_path = './data/checkpoints/last_checkpoint.pth'
model_prefix = ''
if len(sys.argv) == 2:
    model_path = sys.argv[1]
    model_prefix = os.path.splitext(os.path.basename(model_path))[0]

model = AE().cuda()
model.load_state_dict(torch.load(model_path))
model.eval()

latents = inference(X=val_x, model=model)
pred, X_embedded = predict(latents)

val_acc = cal_acc(val_y, pred)

print('\nVal acc: {:.3f}'.format(val_acc))