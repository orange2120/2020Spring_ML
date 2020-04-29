import sys, os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time
from myModel import Classifier, ImgDataset

batch_size = 128

dataset_path = '../data/food-11/data.npz'
model_path = './data/model.pkl'
output_path = './output/predict.csv'

if len(sys.argv) == 3:
    model_path = sys.argv[1] + '/model.pkl'
    output_path = sys.argv[2]

elif len(sys.argv) == 2:
    model_path = sys.argv[1] 
    output_path = './output/predict_{}.csv'.format(os.path.splitext(os.path.basename(model_path))[0])

loadfile = np.load(dataset_path)
test_x = loadfile['te_x']

model_best = torch.load(model_path)

test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
])

test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

#將結果寫入 csv 檔
with open(output_path, 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))

print('predict file: {} generated.'.format(output_path))