import h5py
import numpy as np
import pickle as pkl

from scipy.io import loadmat
from sklearn.model_selection import cross_val_score
from sklearn.ensemble.forest import RandomForestClassifier


def load_cars(X_train_path, y_train_path, y_test_path, feature):
    cars_train = h5py.File(X_train_path, 'r')
    cars_train_X = cars_train[feature + '_train']
    cars_test_X = cars_train[feature + '_test']

    cars_train = loadmat(y_train_path)['annotations']
    cars_train_dict = {}
    for i in range(len(cars_train[0])):
        cars_train_dict[cars_train[0][i][5][0]] = cars_train[0][i][4][0][0]
    cars_train_y = []
    for car in sorted(cars_train_dict):
        cars_train_y.append(cars_train_dict[car])
    cars_train_y = np.array(cars_train_y)

    cars_test = loadmat(y_test_path)['annotations']
    cars_test_dict = {}
    for i in range(len(cars_test[0])):
        cars_test_dict[cars_test[0][i][5][0]] = cars_test[0][i][4][0][0]
    cars_test_y = []
    for car in sorted(cars_test_dict):
        cars_test_y.append(cars_test_dict[car])
    cars_test_y = np.array(cars_test_y)

    return cars_train_X, cars_test_X, cars_train_y, cars_test_y


feature = 'fc2'
X_train_path = feature + '_features.h5'
y_train_path = 'cars_train_annos.mat'
y_test_path = 'cars_test_annos_withlabels.mat'

# Loading the cars dataset features
cars_train_X, cars_test_X, cars_train_y, cars_test_y = load_cars(X_train_path, y_train_path, y_test_path, feature)
cars_train_X = np.asarray(cars_train_X).reshape(cars_train_X.shape[0], np.prod(cars_train_X.shape[1:]))
cars_test_X = np.asarray(cars_test_X).reshape(cars_test_X.shape[0], np.prod(cars_test_X.shape[1:]))

cls = RandomForestClassifier(n_estimators=1000, verbose=True, n_jobs=4)
# cls.fit(cars_train_X, cars_train_y)

scores = cross_val_score(cls, cars_train_X, cars_train_y, n_jobs=4, cv=5, verbose=True)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

with open('models/4096_random_forest.pkl', 'wb') as f:
    pkl.dump(cls, f)


