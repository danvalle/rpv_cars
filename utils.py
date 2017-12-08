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
