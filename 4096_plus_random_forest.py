import pickle as pkl

import numpy as np
from sklearn.ensemble.forest import RandomForestClassifier

from utils import load_cars

# only care for fc2 this time
feature = 'fc2'
X_train_path = feature + '_features.h5'
y_train_path = 'cars_train_annos.mat'
y_test_path = 'cars_test_annos_withlabels.mat'

# Loading the cars dataset features
cars_train_X, cars_test_X, cars_train_y, cars_test_y = load_cars(X_train_path, y_train_path, y_test_path, feature)
cars_train_X = np.asarray(cars_train_X).reshape(cars_train_X.shape[0], np.prod(cars_train_X.shape[1:]))
cars_test_X = np.asarray(cars_test_X).reshape(cars_test_X.shape[0], np.prod(cars_test_X.shape[1:]))

cls = RandomForestClassifier(n_estimators=100, verbose=True, n_jobs=4)
cls.fit(cars_train_X, cars_train_y)
# classifier is random forest, if desired to use bagging with 5 subsets, remove the comments below
# the argument max_features=0.2 splits the features in 1/5 of the total features
# cls = RandomForestClassifier(n_estimators=100, verbose=True, n_jobs=4, max_features=0.2)


# if desired to show accuracy and std-dev values for the training set, uncomment the 3 lines below
# scores = cross_val_score(cls, cars_train_X, cars_train_y, cv=5, verbose=True)
# print(scores)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# save predictions for plotting the heatmap
preds = cls.predict(cars_test_X)

with open('models/4096_random_forest.sav', 'wb') as f:
    pkl.dump((preds, cars_test_y), f)

