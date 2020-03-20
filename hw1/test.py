import sys
import pandas as pd
import numpy as np
import csv

npzfile = np.load('save.npz')
mean_x = npzfile['arr_0']
std_x = npzfile['arr_1']
# print(mean_x)
# print(std_x)

# testdata = pd.read_csv('gdrive/My Drive/hw1-regression/test.csv', header = None, encoding = 'big5')
testdata = pd.read_csv('./test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()

print(test_data.shape)

test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    # test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
    test_x[i, :] = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
print(test_x.shape)
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
# print(test_x)

w = np.load('weight.npy')
ans_y = np.dot(test_x, w)
# print(ans_y)

with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)
