import h5py
import numpy as np

from scipy.io import loadmat
from sklearn.model_selection import train_test_split

# utility function to downsample dataset
def cars_downsample(labels_path, n):
    cars_train = loadmat(labels_path)['annotations']
    cars_train_dict = {}
    for i in range(len(cars_train[0])):
        if cars_train[0][i][4][0][0] not in cars_train_dict:
            cars_train_dict[cars_train[0][i][4][0][0]] = [cars_train[0][i][5][0]]
        else:
            cars_train_dict[cars_train[0][i][4][0][0]].append(cars_train[0][i][5][0])
    return zip(*[(cars, cars_train_dict[cars][:40]) for cars in cars_train_dict.keys() if len(cars_train_dict[cars]) >= 40][:25])

# utility function to split the features in h5
def split_dataset(X_path, feature, test_size):
    cars = h5py.File(X_path, 'r')
    cars_X = np.array(cars[feature])
    cars_y = np.array(cars[feature + '_labels'])
    return train_test_split(cars_X, cars_y, test_size=test_size, random_state=42)

def split_dataset_fusions(test_size):
    cars_c1_h5 = h5py.File('c1_features.h5', 'r')
    cars_X_c1 = np.array(cars_c1_h5['c1'])
    cars_y_c1 = np.array(cars_c1_h5['c1_labels'])
    cars_c5_h5 = h5py.File('c5_features.h5', 'r')
    cars_X_c5 = np.array(cars_c5_h5['c5'])
    cars_y_c5 = np.array(cars_c5_h5['c5_labels'])
    cars_fc2_h5 = h5py.File('fc2_features.h5', 'r')
    cars_X_fc2 = np.array(cars_fc2_h5['fc2'])
    cars_y_fc2 = np.array(cars_fc2_h5['fc2_labels'])

    # shuffled array
    shuffled_X = np.arange(cars_X_c1.shape[0])
    shuffled_y = np.arange(cars_y_c1.shape[0])
    indices_X_train, indices_X_test, indices_y_train, indices_y_test = train_test_split(shuffled_X, shuffled_y, test_size=test_size, random_state=42)
    
    cars_c1 = cars_X_c1[indices_X_train], cars_X_c1[indices_X_test], cars_y_c1[indices_y_train], cars_y_c1[indices_y_test]
    cars_c5 = cars_X_c5[indices_X_train], cars_X_c5[indices_X_test], cars_y_c5[indices_y_train], cars_y_c5[indices_y_test]
    cars_fc2 = cars_X_fc2[indices_X_train], cars_X_fc2[indices_X_test], cars_y_fc2[indices_y_train], cars_y_fc2[indices_y_test]

    return cars_c1, cars_c5, cars_fc2
