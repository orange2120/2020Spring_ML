# main_test.py
import os, sys
import numpy as np
import pandas as pd
import torch
import time
from preprocess import Preprocess
from data import TwitterDataset
from test import testing
import utils as u

sen_len = 40
batch_size = 128

model_path = './data/ckpt.model'

path_prefix = './data/'
output_prefix = './output/'

if len(sys.argv) == 2:
    model_path = sys.argv[1]

testing_data = os.path.join(path_prefix, 'testing_data.txt')
model_dir = os.path.join(path_prefix, 'model/') # model directory for checkpoint model
w2v_path = os.path.join(path_prefix, 'model/w2v_all.model') # 處理word to vec model的路徑

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 開始測試模型並做預測
print("loading testing data ...")
test_x = u.load_testing_data(testing_data)
preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2idx()
test_dataset = TwitterDataset(X=test_x, y=None)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)
print('\nload model ...')
model = torch.load(model_path)
outputs = testing(batch_size, test_loader, model, device)

# 寫到csv檔案供上傳kaggle
tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
print("save csv ...")

output_name = os.path.splitext(os.path.basename(model_path))[0]

tmp.to_csv(os.path.join(output_prefix, 'predict_{}.csv'.format(output_name)), index=False)
print("Finish Predicting")
