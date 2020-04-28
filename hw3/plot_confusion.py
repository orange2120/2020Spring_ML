import sys, os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from myModel import Classifier, ImgDataset
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./figure/confusion_matrix.png')
    plt.show()


model_path = './data/model_20200410-02-35-27.pkl'
dataset_path = '../data/food-11/data.npz'

loadfile = np.load(dataset_path)
val_x = loadfile['val_x']
val_y = loadfile['val_y']

model_best = torch.load(model_path)

test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
])

val_set = ImgDataset(val_x, val_y, test_transform)
test_set = ImgDataset(val_x, transform=test_transform)

batch_size = 1
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

print(val_y)
print(val_y.shape)

model_best.eval()
# pred_y = np.empty(val_y.shape, dtype=np.uint8)
pred_y = np.empty(val_y.shape, dtype=int)
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1).astype(np.uint8)
        # print(test_label.dtype)
        # for y in test_label:
        # np.append(pred_y, test_label)
        pred_y[i] = test_label

print(pred_y)
print(pred_y.shape)

cm = confusion_matrix(val_y, pred_y)

names = ('Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food'
, 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit')

plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, names)

