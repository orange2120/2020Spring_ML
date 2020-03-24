import sys
import numpy as np
import matplotlib.pyplot as plt
import util as u
from keras.models import Sequential
from keras.layers import Dense

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

model = Sequential()
model.add(Dense(12, input_dim=40, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=20, batch_size=8)

acc = model.evaluate(X_train, Y_train)
print('Accu: %.2f' % (acc * 100))