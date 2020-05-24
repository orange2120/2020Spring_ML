import re
import torch
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

dataset_dir = './data'

class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, folderName, transform=None):
        self.transform = transform
        self.data = []
        self.label = []

        for img_path in glob(folderName + '/*.jpg'):
            try:
                # Get classIdx by parsing image path
                class_idx = int(re.findall(re.compile(r'\d+'), img_path)[1])
            except:
                # if inference mode (there's no answer), class_idx default 0
                class_idx = 0

            image = Image.open(img_path)

            # Get File Descriptor
            image_fp = image.fp
            image.load()
            # Close File Descriptor (or it'll reach OPEN_MAX)
            image_fp.close()

            self.data.append(image)
            self.label.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.label[idx]


trainTransform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
testTransform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

def get_dataloader(mode='training', batch_size=32):

    assert mode in ['training', 'testing', 'validation']
    '''
    dataset_path = '../data/food-11/data.npz'
    loadfile = np.load(dataset_path)

    if mode == 'training':
        train_x = loadfile['tr_x']
        train_y = loadfile['tr_y']
        dataset = ImgDataset(train_x, train_y, trainTransform)

    elif mode == 'validation':
        val_x = loadfile['val_x']
        val_y = loadfile['val_y']
        dataset = ImgDataset(val_x, val_y, testTransform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training'))

    '''
    dataset = MyDataset(
        f'{dataset_dir}/food-11/{mode}',
        transform=trainTransform if mode == 'training' else testTransform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training'))

    return dataloader


# get dataloader
train_dataloader = get_dataloader('training', batch_size=32)
torch.save(train_dataloader, './data/train_loader.pth')

# valid_dataloader = get_dataloader('validation', batch_size=32)