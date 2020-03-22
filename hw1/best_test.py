import sys
import pandas as pd
import numpy as np
import csv

INPUT_PATH = './best_test.csv'
OUTPUT_PATH = './best_submit.csv'

if len(sys.argv) == 3:
    INPUT_PATH = sys.argv[1]
    OUTPUT_PATH = sys.argv[2]

npzfile = np.load('best_save.npz')
mean_x = npzfile['arr_0']
std_x = npzfile['arr_1']

# testdata = pd.read_csv('gdrive/My Drive/hw1-regression/test.csv', header = None, encoding = 'big5')
testdata = pd.read_csv(INPUT_PATH, header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()

print(test_data.shape) # 4320 (240 * 18 feature) * 9h

test_data = np.reshape(test_data ,(240, -1, 9))
print(test_data.shape)
print(test_data[0].shape)

n = 24

test_feat = np.zeros((240, n, 9), dtype=float) # 
print(test_feat.shape)

# total 240 days  9h of data per day
for day in range(240):
    AMB_TEMP, CH4, CO, NMHC, NO, NO2, NOx, O3, PM10, PM25, RAINFALL, RH, SO2, THC, WD_HR, WIND_DIREC, WIND_SPEED, WS_HR = test_data[day].astype(float)

    test_feat[day] = np.vstack( # should be n features
        [PM25**i for i in range(1,7)]
        +[PM10**i for i in range(1,7)]
        # +[AMB_TEMP**i for i in range(1,2)]
        # +[CH4**i for i in range(1,2)]
        +[CO**i for i in range(1,4)]
        +[NMHC**i for i in range(1,2)]
        +[NO**i for i in range(1,2)]
        +[NO2**i for i in range(1,2)]
        +[NOx**i for i in range(1,2)]
        +[O3**i for i in range(1,2)]
        # +[RAINFALL**i for i in range(1,2)]
        # +[RH**i for i in range(1,2)]
        +[SO2**i for i in range(1,2)]
        # +[THC**i for i in range(1,2)]
        +[WD_HR**i for i in range(1,2)]
        +[WIND_DIREC**i for i in range(1,2)]
        +[WIND_SPEED**i for i in range(1,2)]
        # +[WS_HR**i for i in range(1,2)]
    )

print(test_data.shape)
test_feat = test_feat.reshape(-1, 9)
print(test_data.shape)

# extract n features
test_x = np.empty([240, n*9], dtype = float)
print(test_x.shape)
for i in range(240):
    test_x[i, :] = test_feat[n * i: n * (i + 1), :].reshape(1, -1)

for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
# print(test_x)

w = np.load('best_weight.npy')
ans_y = np.dot(test_x, w)
# print(ans_y)

with open(OUTPUT_PATH, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)
