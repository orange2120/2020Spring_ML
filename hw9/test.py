
# test
import sys, os
import time
import torch
import numpy as np
from preprocess import *
from utils import *
from model import AE

checkpoint_path = './data/checkpoints'
output_dir = './data/output'

def save_prediction(pred, out_csv='prediction.csv'):
    with open(out_csv, 'w') as f:
        f.write('id,label\n')
        for i, p in enumerate(pred):
            f.write(f'{i},{p}\n')
    print(f'Save prediction to {out_csv}.')

print(len(sys.argv))
model_path = f'./{checkpoint_path}/last_checkpoint.pth'
output_path = f'./{output_dir}/prediction.csv'
output_inv_path = f'./{output_dir}/prediction_invert.csv'
if len(sys.argv) == 2:
    model_path = sys.argv[1]
    model_prefix = os.path.splitext(os.path.basename(model_path))[0]
    output_path = f'./{output_dir}/prediction_{model_prefix}.csv'
    output_inv_path = f'./{output_dir}/prediction_{model_prefix}_invert.csv'

# load model
print(f'Load checkpoint: {model_path}')
model = AE().cuda()
model.load_state_dict(torch.load(model_path))
model.eval()

# 準備 data
trainX = np.load('./data/trainX_new.npy')

# 預測答案
latents = inference(X=trainX, model=model)
pred, X_embedded = predict(latents)

# 將預測結果存檔，上傳 kaggle
save_prediction(pred, output_path)

# 由於是 unsupervised 的二分類問題，我們只在乎有沒有成功將圖片分成兩群
# 如果上面的檔案上傳 kaggle 後正確率不足 0.5，只要將 label 反過來就行了
save_prediction(invert(pred), output_inv_path)