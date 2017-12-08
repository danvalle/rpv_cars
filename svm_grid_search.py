import numpy as np
import pickle
import sys

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from utils import split_dataset

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

score = 'accuracy'  # , 'precision', 'recall']

print("# Tuning hyper-parameters for %s" % score)
print()

clf = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, cv=5,
                   scoring=score, n_jobs=8)  # '%s_macro' % score)
clf.fit(cars_train_X, cars_train_y)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = cars_test_y, clf.predict(cars_test_X)
print(classification_report(y_true, y_pred))
print()

print confusion_matrix(y_true, y_pred)

# save model for later use in plotting the heatmap
filename = feature + '_model.sav'
pickle.dump(clf, open(filename, 'wb'))
