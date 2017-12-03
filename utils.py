import h5py
from scipy.io import loadmat
import numpy as np


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
        if len(train_images_to_use) + len(
                to_add) <= 1021:  # chose 1021 to match with number of testing images for classes selected
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
