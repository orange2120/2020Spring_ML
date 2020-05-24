# train
import time
import torch
from torch import optim
import torchvision.transforms as transforms
from preprocess import *
from model import *
from utils import *

checkpoint_path = './data/checkpoints'

n_epoch = 300

same_seeds(0)

model = AE().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

model.train()

trainX = np.load('./data/trainX_new.npy')
trainX_preprocessed = preprocess(trainX)

# img_dataset = Image_Dataset(trainX_preprocessed, train_transform)
img_dataset = Image_Dataset(trainX_preprocessed, None)

# 準備 dataloader, model, loss criterion 和 optimizer
img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)

print('Start training...')

timestr = time.strftime("%Y%m%d-%H-%M-%S")

# 主要的訓練過程
for epoch in range(n_epoch):
    for data in img_dataloader:
        img = data
        img = img.cuda()

        output1, output = model(img)
        # output1, output, mean, logvar = model(img)

        # kl_div = -0.5* torch.sum(logvar + 1-mean**2 - torch.exp(logvar))
        # kl_div = kl_div.sum()/output.shape[0]

        loss = criterion(output, img)
        # loss = criterion(output, img) + kl_div
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f'./{checkpoint_path}/checkpoint_{timestr}_{epoch+1}.pth')
            
    print('epoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, loss.data))

# 訓練完成後儲存 model
torch.save(model.state_dict(), f'./{checkpoint_path}/last_{timestr}_checkpoint.pth')