import numpy as np
import csv

# X_train_fpath = './data/X_train'
# X_train_outpath = './data/X_train_m'
X_train_fpath = './data/X_test'
X_train_outpath = './data/X_test_m'

features = []
ages = [' age_0', ' age_1', ' age_2', ' age_3', ' age_4', ' age_5', ' age_6', ' age_7', ' age_8', ' age_9']
# years   1~10    11~20    21~30    31~40    41~50    51~60    61~70    71~80    81~90    91~

with open(X_train_fpath) as f:
    features = f.readline().strip('\n').split(',')
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
f.close()

features.extend(ages)
print(features)

age_dim_row = len(ages)
data_dim = X_train.shape[0]

print(X_train.shape)

age_table = np.zeros((data_dim ,len(ages)))

# print(X_train[0])

for i in range(data_dim):
    age = X_train[i][0]
    # print(age)
    if   age >= 1 and age <= 10:
        age_table[i][0] = 1
    elif age > 10 and age <= 20:
        age_table[i][1] = 1
    elif age > 20 and age <= 30:
        age_table[i][2] = 1
    elif age > 30 and age <= 40:
        age_table[i][3] = 1
    elif age > 40 and age <= 50:
        age_table[i][4] = 1
    elif age > 50 and age <= 60:
        age_table[i][5] = 1
    elif age > 60 and age <= 70:
        age_table[i][6] = 1
    elif age > 70 and age <= 80:
        age_table[i][7] = 1
    elif age > 80 and age <= 90:
        age_table[i][8] = 1
    elif age > 90:
        age_table[i][9] = 1

new_X_train = np.hstack((X_train, age_table))
print(new_X_train.shape)

with open(X_train_outpath, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(features)
    np.savetxt(f, new_X_train, fmt='%i',delimiter=',')
f.close()