import numpy as np
import pickle
import sys

from cria_heatmap import cria_map
from sklearn.metrics import average_precision_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from utils import split_dataset

# usage: python svm_grid_search.py feature
feature = sys.argv[1]
X_path = feature + '_features.h5'

# Loading the cars dataset features
test_size = 0.3
cars_train_X, cars_test_X, cars_train_y, cars_test_y = split_dataset(X_path, feature, test_size)

cars_train_X = np.asarray(cars_train_X).reshape(cars_train_X.shape[0], np.prod(cars_train_X.shape[1:]))
cars_test_X = np.asarray(cars_test_X).reshape(cars_test_X.shape[0], np.prod(cars_test_X.shape[1:]))

# Set the parameters by cross-validation
tuned_parameters = [{'gamma': [1e-3, 1e-4, 1e-5],
                     'C': [1, 10, 100, 1000]}]

print("# Tuning hyper-parameters for accuracy")
print()

clf = GridSearchCV(SVC(kernel='rbf', cache_size=1024), tuned_parameters, cv=5,
                   scoring='accuracy', n_jobs=-1)
clf.fit(cars_train_X, cars_train_y)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)

# print("Grid scores on development set:")
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))

y_true, y_pred = cars_test_y, clf.predict(cars_test_X)

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

# save model for later use in plotting the heatmap
#filename = feature + '_model.sav'
#pickle.dump(clf, open(filename, 'wb'))
