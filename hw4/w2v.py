# w2v.py
# 這個block是用來訓練word to vector 的 word embedding
# 注意！這個block在訓練word to vector時是用cpu，可能要花到10分鐘以上
import sys, os
import numpy as np
import pandas as pd
import argparse
from gensim.models import word2vec
import utils as u

path_prefix = './data/'
output_model_path = os.path.join(path_prefix, 'model/w2v_all.model')

def train_word2vec(x):
    # 訓練word to vector 的 word embedding
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=1)
    return model

if __name__ == "__main__":
    print("loading training data ...")
    if len(sys.argv) == 3:
        train_x, y = u.load_training_data(sys.argv[1])
        train_x_no_label = u.load_training_data(sys.argv[2])
    else:
        train_x, y = u.load_training_data(os.path.join(path_prefix, 'training_label.txt'))
        train_x_no_label = u.load_training_data(os.path.join(path_prefix, 'training_nolabel.txt'))

    print("loading testing data ...")
    test_x = u.load_testing_data(os.path.join(path_prefix, 'testing_data.txt'))

    print('start training...')
    #model = train_word2vec(train_x + train_x_no_label + test_x)
    model = train_word2vec(train_x + test_x)
    
    print("saving model ...")
    model.save(output_model_path)