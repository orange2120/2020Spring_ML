from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import f1_score, pairwise_distances, roc_auc_score
from scipy.cluster.vq import vq, kmeans
from model import *

from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
import time

# task = 'knn'
task = 'ae'

mode_path = './data/models'
output_path = './data/output'

train = np.load('./data/train.npy', allow_pickle=True)
test = np.load('./data/test.npy', allow_pickle=True)

print('data loaded.')

# print(train.shape)
# for i in range(0, 10):
#     plt.figure()
#     plt.imshow(train[i])
#     plt.show()

if task == 'knn':
    x = train.reshape(len(train), -1)
    y = test.reshape(len(test), -1)
    scores = list()
    for n in range(1, 10):
      kmeans_x = MiniBatchKMeans(n_clusters=n, batch_size=100).fit(x)
      y_cluster = kmeans_x.predict(y)
      y_dist = np.sum(np.square(kmeans_x.cluster_centers_[y_cluster] - y), axis=1)

      y_pred = y_dist
    #   score = f1_score(y_label, y_pred, average='micro')
    #   score = roc_auc_score(y_label, y_pred, average='micro')
    #   scores.append(score)
    # print(np.max(scores), np.argmax(scores))
    # print(scores)
    # print('auc score: {}'.format(np.max(scores)))

if task == 'pca':

    x = train.reshape(len(train), -1)
    y = test.reshape(len(test), -1)
    pca = PCA(n_components=2).fit(x)

    y_projected = pca.transform(y)
    y_reconstructed = pca.inverse_transform(y_projected)  
    dist = np.sqrt(np.sum(np.square(y_reconstructed - y).reshape(len(y), -1), axis=1))
    
    y_pred = dist
    # score = roc_auc_score(y_label, y_pred, average='micro')
    # score = f1_score(y_label, y_pred, average='micro')
    # print('auc score: {}'.format(score))

if task == 'ae':
    num_epochs = 2
    batch_size = 256
    learning_rate = 5e-5

    #{'fcn', 'cnn', 'vae'} 
    model_type = 'fcn' 

    x = train
    if model_type == 'fcn' or model_type == 'vae':
        x = x.reshape(len(x), -1)
        
    data = torch.tensor(x, dtype=torch.float)
    train_dataset = TensorDataset(data)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    model_classes = {'fcn':fcn_autoencoder(), 'cnn':conv_autoencoder(), 'vae':VAE()}
    model = model_classes[model_type].cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate)
    
    best_loss = np.inf
    model.train()

    print('start training...')
    timestr = time.strftime("%Y%m%d-%H-%M-%S")

    for epoch in range(num_epochs):
        for data in train_dataloader:
            if model_type == 'cnn':
                img = data[0].transpose(3, 1).cuda()
            else:
                img = data[0].cuda()
            # ===================forward=====================
            output = model(img)
            if model_type == 'vae':
                loss = loss_vae(output[0], img, output[1], output[2], criterion)
            else:
                loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ===================save====================
            if loss.item() < best_loss:
                best_loss = loss.item()
                # torch.save(model, '{}/best_model_{}.pt'.format(mode_path, model_type))
                torch.save(model, '{}/best_model_{}_{}.pt'.format(mode_path, timestr, model_type))
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))

    # test
    if model_type == 'fcn' or model_type == 'vae':
        # y = test.reshape(len(test_tmp), -1)
        y = test.reshape(len(test), -1)
    else:
        y = test
    
    data = torch.tensor(y, dtype=torch.float)
    test_dataset = TensorDataset(data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    # model = torch.load('{}/best_model_{}.pt'.format(mode_path, model_type), map_location='cuda')
    model = torch.load('{}/best_model_{}_{}.pt'.format(mode_path , timestr, model_type), map_location='cuda')

    model.eval()
    reconstructed = list()
    for i, data in enumerate(test_dataloader): 
        if model_type == 'cnn':
            img = data[0].transpose(3, 1).cuda()
        else:
            img = data[0].cuda()
        output = model(img)
        if model_type == 'cnn':
            output = output.transpose(3, 1)
        elif model_type == 'vae':
            output = output[0]
        reconstructed.append(output.cpu().detach().numpy())

    reconstructed = np.concatenate(reconstructed, axis=0)
    anomality = np.sqrt(np.sum(np.square(reconstructed - y).reshape(len(y), -1), axis=1))
    y_pred = anomality
    # with open(f'{output_path}/prediction.csv', 'w') as f:
    with open(f'{output_path}/prediction_{timestr}.csv', 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(i+1, y_pred[i]))
    # score = roc_auc_score(y_label, y_pred, average='micro')
    # score = f1_score(y_label, y_pred, average='micro')
    # print('auc score: {}'.format(score))


