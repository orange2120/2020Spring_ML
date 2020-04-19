import csv
import numpy as np

out_path = './output/pred_vote_151722.csv'

# input csv file folder
csv_folder = './output/'
# csv filenames used to be voted 
csv_list = [
'predict_ckpt_20200418-15-45-28.csv',
'predict_ckpt_20200418-17-04-42.csv',
'predict_ckpt_20200418-22-58-38.csv'
]

num_classes = 2
num_pred = 200000

pred_cnt = np.zeros((num_pred, num_classes), dtype=int)
print(pred_cnt.shape)

for csvf in csv_list:
    print("Reading: " + csvf)
    with open(csv_folder + csvf) as f:
        reader = csv.reader(f)
        next(reader)

        for index, row in enumerate(reader):
            pred_cnt[index][int(row[1])] = pred_cnt[index][int(row[1])] + 1

with open(out_path, 'w') as f:
    f.write('Id,label\n')
    for i, y in enumerate(pred_cnt):
        f.write('{},{}\n'.format(i, np.argmax(y)))

print('done.')
print('output file: ' + out_path)
