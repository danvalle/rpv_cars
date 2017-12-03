from collections import Counter
import utils

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# manual definition of voting classifier, it takes into account every classifier with weight 1
def vote(all_results):
    final_preds = []
    for i in range(len(all_results[0])):
        counter = Counter([all_results[0][i], all_results[1][i], all_results[2][i]])
        final_preds.append(counter.most_common(1)[0][0])
    return final_preds

features = ['fc2', 'c5', 'fc2']
trained_svms = []
for feature in features:
    X_train_path = feature + '_features.h5'
    y_train_path = 'devkit/cars_train_annos.mat'
    y_test_path = 'devkit/cars_test_annos_withlabels.mat'

    # Loading the cars dataset features
    cars_train_X, cars_test_X, cars_train_y, cars_test_y = utils.load_cars(X_train_path, y_train_path, y_test_path, feature)
    cars_train_X = np.asarray(cars_train_X).reshape(cars_train_X.shape[0], np.prod(cars_train_X.shape[1:]))
    cars_test_X = np.asarray(cars_test_X).reshape(cars_test_X.shape[0], np.prod(cars_test_X.shape[1:]))

    cars_train_X = cars_train_X[:200]
    cars_test_X = cars_test_X[:200]
    cars_train_y = cars_train_y[:200]
    cars_test_y = cars_test_y[:200]

    # To apply an classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = cars_train_y.shape[0] + cars_test_y.shape[0]

    # Set the parameters by cross-validation
    tuned_parameters = [{'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]}]

    score = 'accuracy'
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(kernel='linear'), tuned_parameters, cv=5,
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
    trained_svms.append(np.asarray(y_pred))
    print('preds:', y_pred)

y_pred = vote(trained_svms)
print('final preds:', y_pred)
print(classification_report(y_true, y_pred))
print()
