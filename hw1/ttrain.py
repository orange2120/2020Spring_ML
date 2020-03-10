import sys
import pandas as pd
import numpy as np

data = pd.read_csv('./train.csv', encoding = 'big5')
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

# print(raw_data)

month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample


# extract specific features
for month in range(12):
    # 0       1    2   3     4   5    6    7   8     9     10        11  12   13   14     15          16          17
    AMB_TEMP, CH4, CO, NMHC, NO, NO2, NOx, O3, PM10, PM25, RAINFALL, RH, SO2, THC, WD_HR, WIND_DIREC, WIND_SPEED, WS_HR = month_data[month]
    month_data[month] = np.vstack(
         [PM25**i for i in range(1, 7)] # index 0
        +[PM10**i for i in range(1, 7)]
        +[CO**i   for i in range(1, 2)]
        +[SO2**i  for i in range(1, 2)]
        +[NO**i   for i in range(1, 2)]
        +[SO2**i  for i in range(1, 2)]
        +[O3**i   for i in range(1, 2)]
    )

n = month_data[0].shape[0]
print('n={}'.format(n))

# month_data[month][feature][data per hour]

x = np.empty([12 * 471, n * 9], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            # take all rows 
            # x[month * 471 + day * 24 + hour, 0:] = month_data[month][:2,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            # 0th row,  day * 24 + hour + 9 value
            y[month * 471 + day * 24 + hour, 0] = month_data[month][0, day * 24 + hour + 9] #value
# print(x)
# print(y)

print(x.shape)

mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

dim = n * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
learning_rate = 100
iter_time = 10000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
    if(t%100==0):
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('tweight.npy', w)

# save normalized x data
np.savez('tsave.npz', mean_x, std_x)
# print(mean_x)
# print(std_x)
