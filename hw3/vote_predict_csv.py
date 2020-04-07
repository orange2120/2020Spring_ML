import csv
import numpy as np

out_path = './output/pred_vote_030607.csv'

# input csv file folder
csv_folder = './output/'
# csv filenames used to be voted 
csv_list = [
'predict_model_20200403-02-29-49.csv',
'predict_model_20200406-21-53-19.csv',
'predict_model_20200407-20-03-05.csv']

num_classes = 11
num_pred = 3347

pred_cnt = np.zeros((num_pred, num_classes), dtype=int)
print(pred_cnt.shape)

for csvf in csv_list:
    print("Reading: " + csvf)
    with open(csv_folder + csvf) as f:
        reader = csv.reader(f)
        next(reader)
        for index, row in enumerate(reader):
            pred_cnt[index][int(row[1])] = pred_cnt[index][int(row[1])] + 1
            # index = index + 1

with open(out_path, 'w') as f:
    f.write('Id,Category\n')
    for i, y in enumerate(pred_cnt):
        f.write('{},{}\n'.format(i, np.argmax(y)))

print('done.')
print('output file: ' + out_path)
