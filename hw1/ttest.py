import sys
import pandas as pd
import numpy as np
import csv

npzfile = np.load('tsave.npz')
mean_x = npzfile['arr_0']
std_x = npzfile['arr_1']
# print(mean_x)
# print(std_x)

# testdata = pd.read_csv('gdrive/My Drive/hw1-regression/test.csv', header = None, encoding = 'big5')
testdata = pd.read_csv('./test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()

print(test_data.shape) # 4320 (240 * 18 feature) * 9h
# print(test_data)
test_feat = test_data
test_feat.reshape(240, -1, 9)
print(test_feat.shape)

# total 240 days  9h of data per day
for day in range(240):
    AMB_TEMP, CH4, CO, NMHC, NO, NO2, NOx, O3, PM10, PM25, RAINFALL, RH, SO2, THC, WD_HR, WIND_DIREC, WIND_SPEED, WS_HR = test_data[day]
    test_data[day] = np.vstack(
         [PM25**i for i in range(1, 7)]
        +[PM10**i for i in range(1, 7)]
        +[CO**i   for i in range(1, 2)]
        +[SO2**i  for i in range(1, 2)]
        +[NO**i   for i in range(1, 2)]
        +[SO2**i  for i in range(1, 2)]
        +[O3**i   for i in range(1, 2)]
    )

print(test_data.shape)
test_data = test_data.reshape(-1,9)
print(test_data.shape)

n = 7

# n_dim = test_data[0].shape
print(test_data[0].shape)

# test_x = np.empty([240, 18*9], dtype = float)
test_x = np.empty([240, n*9], dtype = float)
for i in range(240):
    # test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
    test_x[i, :] = test_data[n * i: n * (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
# print(test_x)

w = np.load('tweight.npy')
ans_y = np.dot(test_x, w)
# print(ans_y)

with open('tsubmit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)
