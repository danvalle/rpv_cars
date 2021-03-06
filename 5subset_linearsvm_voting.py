import numpy as np
import pickle as pkl


from cria_heatmap import cria_map
from mlxtend.feature_selection import ColumnSelector
from random import shuffle
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.metrics import average_precision_score, accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.svm.classes import LinearSVC

from utils import split_dataset

feature = 'fc2'
X_path = feature + '_features.h5'

# Loading the cars dataset features
test_size = 0.3
cars_train_X, cars_test_X, cars_train_y, cars_test_y = split_dataset(X_path, feature, test_size)
cars_train_X = np.asarray(cars_train_X).reshape(cars_train_X.shape[0], np.prod(cars_train_X.shape[1:]))
cars_test_X = np.asarray(cars_test_X).reshape(cars_test_X.shape[0], np.prod(cars_test_X.shape[1:]))

# randomly split features in 5 subsets of (roughly) the same size
splits_cols = [z for z in range(cars_train_X.shape[-1])]
shuffle(splits_cols)
sz = int(len(splits_cols) / 5)
split1 = splits_cols[:sz]
split2 = splits_cols[sz:sz*2]
split3 = splits_cols[sz*2:sz*3]
split4 = splits_cols[sz*3:sz*4]
split5 = splits_cols[sz*4:]

# create a pipeline meta-classifier (sklearn) and use a linearsvc at the end as the classifier
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

# create the ensemble with the votingclassifier
cls = VotingClassifier([
    ('l1', pipe1),
    ('l2', pipe2),
    ('l3', pipe3),
    ('l4', pipe4),
    ('l5', pipe5),
], n_jobs=4)
cls.fit(cars_train_X, cars_train_y)

# uncomment the 3 lines below if needed to see the accuracy and std-dev of the training set
# scores = cross_val_score(cls, cars_train_X, cars_train_y, cv=5, verbose=True)
# print(scores)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# this reaches about 30% acc

# create the predictions and dump to a file for plotting the heatmap
y_pred = cls.predict(cars_test_X)

with open('5subset_linearsvm_voting.sav', 'wb') as f:
    pkl.dump((y_pred, cars_test_y), f)

y_true = cars_test_y
preds = {}
for i in range(y_true.shape[0]):
  if y_true[i] == y_pred[i]:
    if y_true[i] not in preds:
      preds[y_true[i]] = 1
    else:
      preds[y_true[i]] += 1
for y in y_true:
  if y not in preds:
    preds[y] = 0

conf_matrix = 1. * confusion_matrix(y_true, y_pred)
conf_matrix /= np.sum(conf_matrix, axis=1)
cria_map(conf_matrix, preds.keys(), feature)

print "OA:", accuracy_score(y_true, y_pred)
print "AA:", np.trace(conf_matrix) / (1. * conf_matrix.shape[0])



