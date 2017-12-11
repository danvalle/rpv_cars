import pickle as pkl
import sys
import numpy as np

from cria_heatmap import cria_map
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics import average_precision_score, accuracy_score, classification_report, confusion_matrix

from utils import split_dataset

# only care for fc2 this time
feature = 'fc2'
X_path = feature + '_features.h5'

# Loading the cars dataset features
test_size = 0.3
cars_train_X, cars_test_X, cars_train_y, cars_test_y = split_dataset(X_path, feature, test_size)
cars_train_X = np.asarray(cars_train_X).reshape(cars_train_X.shape[0], np.prod(cars_train_X.shape[1:]))
cars_test_X = np.asarray(cars_test_X).reshape(cars_test_X.shape[0], np.prod(cars_test_X.shape[1:]))

#cls = RandomForestClassifier(n_estimators=100, verbose=True, n_jobs=-1)
# classifier is random forest, if desired to use bagging with 5 subsets, remove the comments below
# the argument max_features=0.2 splits the features in 1/5 of the total features
cls = RandomForestClassifier(n_estimators=100, verbose=True, n_jobs=-1, max_features=0.2)
cls.fit(cars_train_X, cars_train_y)


# if desired to show accuracy and std-dev values for the training set, uncomment the 3 lines below
# scores = cross_val_score(cls, cars_train_X, cars_train_y, cv=5, verbose=True)
# print(scores)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# save predictions for plotting the heatmap
y_pred = cls.predict(cars_test_X)

with open('bagging.sav', 'wb') as f:
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
cria_map(conf_matrix, preds.keys(), 'bagging')

print "OA:", accuracy_score(y_true, y_pred)
print "AA:", np.trace(conf_matrix) / (1. * conf_matrix.shape[0])

