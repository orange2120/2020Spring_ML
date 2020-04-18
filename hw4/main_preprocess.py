
import os, sys
import torch
from preprocess import Preprocess
import utils as u

# 定義句子長度
# sen_len = 20
sen_len = 40

path_prefix = './data/'

# 處理好各個data的路徑
train_with_label = os.path.join(path_prefix, 'training_label.txt')
train_no_label = os.path.join(path_prefix, 'training_nolabel.txt')
w2v_path = os.path.join(path_prefix, 'model/w2v_all.model') # 處理word to vec model的路徑


print("loading data ...") # 把'training_label.txt'跟'training_nolabel.txt'讀進來
train_x, y = u.load_training_data(train_with_label)
train_x_no_label = u.load_training_data(train_no_label)

# 對input跟labels做預處理
preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(y)

# 把data分為training data跟validation data(將一部份training data拿去當作validation data)
X_train, X_val, y_train, y_val = train_x[:180000], train_x[180000:], y[:180000], y[180000:]

print("saving processed data ...")
torch.save(X_train, os.path.join(path_prefix, 'X_train.pt'))
torch.save(X_val, os.path.join(path_prefix, 'X_val.pt'))
torch.save(y_train, os.path.join(path_prefix, 'y_train.pt'))
torch.save(y_val, os.path.join(path_prefix, 'y_val.pt'))
torch.save(embedding, os.path.join(path_prefix, 'embedding.pt'))
print('done.')