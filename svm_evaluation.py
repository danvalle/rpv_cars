import pickle
import utils

import numpy as np

# only care FC2 for this time
feature = 'fc2'
X_train_path = feature + '_features.h5'
y_train_path = 'cars_train_annos.mat'
y_test_path = 'cars_test_annos_withlabels.mat'

# Loading the cars dataset features
cars_train_X, cars_test_X, cars_train_y, cars_test_y, used_classes = utils.load_cars(X_train_path, y_train_path, y_test_path, feature)
cars_train_X = np.asarray(cars_train_X).reshape(cars_train_X.shape[0], np.prod(cars_train_X.shape[1:]))
cars_test_X = np.asarray(cars_test_X).reshape(cars_test_X.shape[0], np.prod(cars_test_X.shape[1:]))

filename = feature + '_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
y_pred = loaded_model.predict(cars_test_X)
score = loaded_model.score(cars_test_X, cars_test_y)
print('OA:', score)

# evaluate overall and average acuracy for the svm
classes_acc_sum = 0
for class_ in used_classes:
    classes_acc_sum += np.sum(y_pred[np.where(cars_test_y == class_)] == class_) / (1.0 * np.sum(cars_test_y == class_))
print('AA:', classes_acc_sum / len(used_classes))

filename = feature + '_predictions_a.sav'
pickle.dump([y_pred, cars_test_y], open(filename, 'wb'))