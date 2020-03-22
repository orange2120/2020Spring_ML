import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

features = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM25', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR' ]

data = pd.read_csv('./train.csv', encoding = 'big5')
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

# print(type(data))
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample

print(month_data[0].shape)
print(month_data[0][9].shape)
'''
squ = []
for i in range(0, 9):
    squ.append(month_data[0][i] ** 2)
for i in range(10, 18):
    squ.append(month_data[0][i] ** 2)
'''
# print(squ)



x = month_data[0][9]
for i in range(0, 9):
    plt.title('PM2.5 - '+features[i])
    plt.xlabel('PM2.5')
    plt.ylabel(features[i])
    print('index={}'.format(i))
    plt.scatter(x, month_data[0][i], s=5)
    plt.savefig('./features/'+ str(features[i]) + '.png')
    plt.clf()
    # plt.show()

for i in range(10, 18):
    plt.title('PM2.5 - '+features[i])
    plt.xlabel('PM2.5')
    plt.ylabel(features[i])
    print('index={}'.format(i))
    plt.scatter(x, month_data[0][i], s=5)
    plt.savefig('./features/'+ str(features[i]) + '.png')
    plt.clf()

    # plt.show()

    # y = month_data[0][1]

# for i in range(len(squ)):
#     print('index={}'.format(i))
#     plt.scatter(x, squ[i])
#     plt.show()
