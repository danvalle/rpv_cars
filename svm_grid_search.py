import h5py
import numpy as np
import pickle
import sys

from scipy.io import loadmat
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

def load_cars(X_train_path, y_train_path, y_test_path, feature):
    cars_train = h5py.File(X_train_path, 'r')
    cars_train_X = cars_train[feature + '_train']
    cars_test_X = cars_train[feature + '_test']
    
    cars_train = loadmat(y_train_path)['annotations']
    cars_train_dict = {}
    selection_train_dict = {}
    
    # Getting 100 training image features by class
    for i in range(len(cars_train[0])):    	
    	cars_train_dict[cars_train[0][i][5][0]] = cars_train[0][i][4][0][0]

    	if cars_train[0][i][4][0][0] not in selection_train_dict:
    		selection_train_dict[cars_train[0][i][4][0][0]] = [cars_train[0][i][5][0]]
    	elif len(selection_train_dict[cars_train[0][i][4][0][0]]) < 100:
        	selection_train_dict[cars_train[0][i][4][0][0]].append(cars_train[0][i][5][0])
    
    used_classes = []
    train_images_to_use = []
    for class_ in selection_train_dict:
    	to_add = selection_train_dict[class_]
    	if len(train_images_to_use) + len(to_add) <= 1021: # chose 1021 to match with number of testing images for classes selected
    		train_images_to_use += to_add
    		used_classes.append(class_)
    	else:
    		train_images_to_use += to_add[:-(len(train_images_to_use) + len(to_add) - 1021)]
    		used_classes.append(class_)
    		break

    cars_train_y = []
    cars_train_X_selected = []
    for car in train_images_to_use:
        cars_train_y.append(cars_train_dict[car])  
        cars_train_X_selected.append(cars_train_X[int(car.split('.')[0]) - 1])  
    cars_train_y = np.array(cars_train_y)

    cars_test = loadmat(y_test_path)['annotations']
    cars_test_dict = {}
    selection_test_dict = {}

    # Getting 100 testing image features by class
    for i in range(len(cars_test[0])):
        cars_test_dict[cars_test[0][i][5][0]] = cars_test[0][i][4][0][0]

        if cars_test[0][i][4][0][0] in used_classes:
        	if cars_test[0][i][4][0][0] not in selection_test_dict:
    			selection_test_dict[cars_test[0][i][4][0][0]] = [cars_test[0][i][5][0]]
    		elif len(selection_test_dict[cars_test[0][i][4][0][0]]) < 100:
        		selection_test_dict[cars_test[0][i][4][0][0]].append(cars_test[0][i][5][0])

    test_images_to_use = []
    for class_ in used_classes:
    	test_images_to_use += selection_test_dict[class_]

    test_images_to_use = sorted(test_images_to_use)

    cars_test_y = []
    cars_test_X_selected = []
    for car in test_images_to_use:
        cars_test_y.append(cars_test_dict[car]) 
        cars_test_X_selected.append(cars_test_X[int(car.split('.')[0]) - 1])   
    cars_test_y = np.array(cars_test_y)  	

    cars_train_X = np.array(cars_train_X_selected)
    cars_test_X = np.array(cars_test_X_selected)
    
    return cars_train_X, cars_test_X, cars_train_y, cars_test_y


feature = 'c5'
X_train_path = feature + '_features.h5'
y_train_path = 'cars_train_annos.mat'
y_test_path = 'cars_test_annos_withlabels.mat'

# Loading the cars dataset features
cars_train_X, cars_test_X, cars_train_y, cars_test_y = load_cars(X_train_path, y_train_path, y_test_path, feature)
cars_train_X = np.asarray(cars_train_X).reshape(cars_train_X.shape[0], np.prod(cars_train_X.shape[1:]))
cars_test_X = np.asarray(cars_test_X).reshape(cars_test_X.shape[0], np.prod(cars_test_X.shape[1:]))

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = cars_train_y.shape[0] + cars_test_y.shape[0]

# Set the parameters by cross-validation
tuned_parameters = [{'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]

score = 'accuracy'#, 'precision', 'recall']

print("# Tuning hyper-parameters for %s" % score)
print()

clf = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, cv=5,
                   scoring=score, n_jobs=8)#'%s_macro' % score)
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

filename = feature + '_model.sav'
pickle.dump(clf, open(filename, 'wb'))