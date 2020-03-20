import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv('./train.csv', encoding = 'big5')

data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample
# feature
for month in range(12):
    AMB_TEMP, CH4, CO, NMHC, NO, NO2, NOx, O3, PM10, PM25, RAINFALL, RH, SO2, THC, WD_HR, WIND_DIREC, WIND_SPEED, WS_HR = month_data[month]
    WIND_DIREC_X, WIND_DIREC_Y = np.cos(WIND_DIREC*np.pi/180)*WIND_SPEED, np.sin(WIND_DIREC*np.pi/180)*WIND_SPEED
    month_data[month] = np.vstack(
        [PM25**i for i in range(1,7)]
        +[PM10**i for i in range(1,7)]
        # +[AMB_TEMP**i for i in range(1,2)]
        +[CH4**i for i in range(1,2)]
        +[CO**i for i in range(1,2)]
        +[NMHC**i for i in range(1,2)]
        +[NO**i for i in range(1,2)]
        +[NO2**i for i in range(1,2)]
        +[NOx**i for i in range(1,2)]
        +[O3**i for i in range(1,2)]
        +[RAINFALL**i for i in range(1,2)]
        +[RH**i for i in range(1,2)]
        +[SO2**i for i in range(1,2)]
        +[THC**i for i in range(1,2)]
        +[WD_HR**i for i in range(1,2)]
        +[WIND_DIREC_X**i for i in range(1,2)]
        +[WIND_DIREC_Y**i for i in range(1,2)]
        +[WIND_SPEED**i for i in range(1,2)]
        +[WS_HR**i for i in range(1,2)]
    )

n = month_data[0].shape[0]
print('n={}'.format(n))

x = np.empty([12 * 471, n * 9], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][0 , day * 24 + hour + 9] #value

# normalize
mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]


dim = n * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
# learning_rate = 100
# iter_time = 10000
# adagrad = np.zeros([dim, 1])
# eps = 0.0000000001
# for t in range(iter_time):
#     loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
#     if(t%100==0):
#         print(str(t) + ":" + str(loss))
#     gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
#     adagrad += gradient ** 2
#     w = w - learning_rate * gradient / np.sqrt(adagrad + eps)

# np.save('weight.npy', w)

print(x.shape, y.shape)
# p = np.random.permutation(len(x))
# x, y = x[p], y[p]
x_train, x_val = x[:5000], x[5000:]
y_train, y_val = y[:5000], y[5000:]
reg = LinearRegression().fit(x_train, y_train)
loss = np.sqrt(np.sum(np.power(reg.predict(x_train) - y_train, 2))/len(y_train))
print(loss)
loss = np.sqrt(np.sum(np.power(reg.predict(x_val) - y_val, 2))/len(y_val))
print(loss)

testdata = pd.read_csv('./test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy().astype(float)
# feature
print(test_data.shape)

test_data = test_data.reshape(240,-1,9) # convert to 240 18 9
print(test_data.shape)

# test_feat 

test_feat = np.empty([240,n,9], dtype=float)
print(test_feat.shape)
for i in range(240):
    AMB_TEMP, CH4, CO, NMHC, NO, NO2, NOx, O3, PM10, PM25, RAINFALL, RH, SO2, THC, WD_HR, WIND_DIREC, WIND_SPEED, WS_HR = test_data[i]
    WIND_DIREC_X, WIND_DIREC_Y = np.cos(WIND_DIREC*np.pi/180)*WIND_SPEED, np.sin(WIND_DIREC*np.pi/180)*WIND_SPEED
    test_feat[i] = np.vstack(
        [PM25**i for i in range(1,11)]
        +[PM10**i for i in range(1,11)]
        +[AMB_TEMP**i for i in range(1,11)]
        +[CH4**i for i in range(1,11)]
        +[CO**i for i in range(1,11)]
        +[NMHC**i for i in range(1,11)]
        +[NO**i for i in range(1,11)]
        +[NO2**i for i in range(1,11)]
        +[NOx**i for i in range(1,11)]
        +[O3**i for i in range(1,11)]
        +[RAINFALL**i for i in range(1,11)]
        +[RH**i for i in range(1,11)]
        +[SO2**i for i in range(1,11)]
        +[THC**i for i in range(1,11)]
        +[WD_HR**i for i in range(1,11)]
        +[WIND_DIREC_X**i for i in range(1,11)]
        +[WIND_DIREC_Y**i for i in range(1,11)]
        +[WIND_SPEED**i for i in range(1,11)]
        +[WS_HR**i for i in range(1,11)]
    )
test_feat = test_feat.reshape(-1,9)

# test_x = np.empty([240, n*9], dtype = float)
# for i in range(240):
#     test_x[i, :] = test_feat[n * i: n* (i + 1), :].reshape(1, -1)
# for i in range(len(test_x)):
#     for j in range(len(test_x[0])):
#         if std_x[j] != 0:
#             test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
# test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

# # w = np.load('weight.npy')
# # ans_y = np.dot(test_x, w)
# ans_y = reg.predict(test_x)

# import csv
# with open('submit.csv', mode='w', newline='') as submit_file:
#     csv_writer = csv.writer(submit_file)
#     header = ['id', 'value']
#     csv_writer.writerow(header)
#     for i in range(240):
#         row = ['id_' + str(i), ans_y[i][0]]
#         csv_writer.writerow(row)
