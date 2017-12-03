import pickle as pkl
from random import shuffle

import numpy as np
from mlxtend.feature_selection import ColumnSelector
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.svm.classes import LinearSVC

from rpv_cars.utils import load_cars

feature = 'fc2'
X_train_path = feature + '_features.h5'
y_train_path = 'cars_train_annos.mat'
y_test_path = 'cars_test_annos_withlabels.mat'

# Loading the cars dataset features
cars_train_X, cars_test_X, cars_train_y, cars_test_y = load_cars(X_train_path, y_train_path, y_test_path, feature)
cars_train_X = np.asarray(cars_train_X).reshape(cars_train_X.shape[0], np.prod(cars_train_X.shape[1:]))
cars_test_X = np.asarray(cars_test_X).reshape(cars_test_X.shape[0], np.prod(cars_test_X.shape[1:]))

splits_cols = [z for z in range(cars_train_X.shape[-1])]
shuffle(splits_cols)
sz = int(len(splits_cols) / 5)
split1 = splits_cols[:sz]
split2 = splits_cols[sz:sz*2]
split3 = splits_cols[sz*2:sz*3]
split4 = splits_cols[sz*3:sz*4]
split5 = splits_cols[sz*4:]

pipe1 = make_pipeline(ColumnSelector(cols=split1),
                      LinearSVC())
pipe2 = make_pipeline(ColumnSelector(cols=split2),
                      LinearSVC())
pipe3 = make_pipeline(ColumnSelector(cols=split3),
                      LinearSVC())
pipe4 = make_pipeline(ColumnSelector(cols=split4),
                      LinearSVC())
pipe5 = make_pipeline(ColumnSelector(cols=split5),
                      LinearSVC())

cls = VotingClassifier([
    ('l1', pipe1),
    ('l2', pipe2),
    ('l3', pipe3),
    ('l4', pipe4),
    ('l5', pipe5),
], n_jobs=4)
cls.fit(cars_train_X, cars_train_y)

# scores = cross_val_score(cls, cars_train_X, cars_train_y, cv=5, verbose=True)
# print(scores)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

preds = cls.predict(cars_test_X)

with open('models/5subset_linearsvm_voting.sav', 'wb') as f:
    pkl.dump((preds, cars_test_y), f)

