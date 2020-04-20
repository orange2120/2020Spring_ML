import sys, os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time
from myModel import Classifier, ImgDataset

batch_size = 128

output_name = 'w0307081007'

model_dir = './data/'
model_path = [
'model_20200403-02-29-49.pkl',
'model_20200407-20-03-05.pkl',
'model_20200408-20-49-52.pkl',
'model_20200410-02-35-27.pkl',
'model_res_20200407-10-05-39.pkl'
]

loadfile = np.load('../data/food-11/data.npz')
test_x = loadfile['te_x']

test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
])
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

models = []
for m in model_path:
    print('Loading model ' + m)
    model = torch.load(model_dir + m)
    models.append(model)
    model.eval()

prediction = []

with torch.no_grad():
    for i, data in enumerate(test_loader):
        testm = models[0](data.cuda())
        for m in range(1, len(models)):
            testm.add(models[m](data.cuda()))
            # test_shape = testm.cpu().data.numpy().shape
            # np.add(test_pred, testm.cpu().data.numpy())
        # print
        # test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        # for y in test_label:
        #     prediction.append(y)

#將結果寫入 csv 檔
'''
with open('./output/predict_vote_{}.csv'.format(output_name), 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))

print('predict file: predict_vote_{}.csv generated.'.format(output_name))
'''