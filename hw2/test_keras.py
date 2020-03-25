import sys
import numpy as np
from keras.models import load_model

model_path = './weight.h5'
output_fpath = './output_{}.csv'

if len(sys.argv) == 2:
    output_fpath = sys.argv[1]

X_test = np.load('save_keras.npy')

model = load_model(model_path)

# predictions = model.predict(X_test)

# print(predictions)

prediction_class = model.predict_classes(X_test)
# prediction_class.flatten()
# print(prediction_class)
print(prediction_class.shape)

with open(output_fpath.format('keras'), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(prediction_class):
        f.write('{},{}\n'.format(i, label[0]))