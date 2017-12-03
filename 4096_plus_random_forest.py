import numpy as np
import pickle as pkl

from sklearn.ensemble.forest import RandomForestClassifier
import pickle as pkl

import numpy as np
from sklearn.ensemble.forest import RandomForestClassifier

from rpv_cars.utils import load_cars

feature = 'fc2'
X_train_path = feature + '_features.h5'
y_train_path = 'cars_train_annos.mat'
y_test_path = 'cars_test_annos_withlabels.mat'

# Loading the cars dataset features
cars_train_X, cars_test_X, cars_train_y, cars_test_y = load_cars(X_train_path, y_train_path, y_test_path, feature)
cars_train_X = np.asarray(cars_train_X).reshape(cars_train_X.shape[0], np.prod(cars_train_X.shape[1:]))
cars_test_X = np.asarray(cars_test_X).reshape(cars_test_X.shape[0], np.prod(cars_test_X.shape[1:]))

cls = RandomForestClassifier(n_estimators=10, verbose=True, n_jobs=4)
cls.fit(cars_train_X, cars_train_y)

# cls.fit(cars_train_X, cars_train_y)

# scores = cross_val_score(cls, cars_train_X, cars_train_y, cv=5, verbose=True)
# print(scores)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

preds = cls.predict(cars_test_X)

with open('models/4096_random_forest.sav', 'wb') as f:
    pkl.dump((preds, cars_test_y), f)

