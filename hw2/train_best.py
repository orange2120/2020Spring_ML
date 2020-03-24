import sys
import numpy as np
import matplotlib.pyplot as plt
import util as u
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.regularizers import l1, L1L2

TRAIN_EPOCHES = 20
TRAIN_BATCH_SIZE = 16

np.random.seed(0)
X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'

if len(sys.argv) == 4:
    X_train_fpath = sys.argv[3]
    Y_train_fpath = sys.argv[4]
    X_test_fpath = sys.argv[5]

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

def _train_dev_split(X, Y, dev_ratio = 0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

print('Splitting data...\n')

# Normalize training and testing data
X_train, X_mean, X_std = u._normalize(X_train, train = True)
X_test, _, _= u._normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)

dev_ratio = 0.1
# X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)

np.save('save_best.npy', X_test)

model = Sequential()
model.add(Dense(120, input_dim=510, activation='relu'))
model.add(Dropout(0.2))
# model.add(Dense(60, activation='relu', activity_regularizer=l1(0.001)))
model.add(Dense(60, activation='relu', kernel_regularizer=L1L2(l1=0.0, l2=0.1)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_split=dev_ratio, epochs=TRAIN_EPOCHES, batch_size=TRAIN_BATCH_SIZE)

_, acc = model.evaluate(X_train, Y_train)
# _, dev_acc = model.evaluate(X_dev, Y_dev)

print('Train acc: %.2f' % (acc * 100))
# print('Dev acc: %.2f' % (dev_acc * 100))
# print(acc)

model.save('./weight.h5')
