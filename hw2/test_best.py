import sys
import numpy as np
from keras.models import load_model

# X_test_fpath = './data/X_test'
output_fpath = './output_{}.csv'

X_test = np.load('save_best.npy')

model = load_model('./weight.h5')

# predictions = model.predict(X_test)

# print(predictions)

prediction_class = model.predict_classes(X_test)
# prediction_class.flatten()
# print(prediction_class)
print(prediction_class.shape)

with open(output_fpath.format('best'), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(prediction_class):
        f.write('{},{}\n'.format(i, label[0]))