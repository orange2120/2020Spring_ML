import glob, sys, os
import torch
import numpy as np
from model import AE
from utils import *

# os.chdir("./data/checkpoints")

keyword = sys.argv[1]
checkpoints = sorted(glob.glob(f'./data/checkpoints/*{keyword}*'))

val_x = np.load('./data/valX.npy')
val_y = np.load('./data/valY.npy')

for file in checkpoints:
    model_path = f'{file}'
    model = AE().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    latents = inference(X=val_x, model=model)
    pred, X_embedded = predict(latents)

    val_acc = cal_acc(val_y, pred)

    print('\n{} | Val acc: {:.3f}'.format(file, val_acc))

# print(type(checkpoints))
#for file in glob.glob("*0520*"):
#    print(file)