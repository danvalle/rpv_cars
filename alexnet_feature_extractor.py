import h5py
import numpy as np
import os
import sys

from glob import glob
from PIL import Image
from scipy.io import loadmat
from torch import cat
from torch.autograd import Variable
from torch.nn import Sequential
from torchvision import transforms
from torchvision.models import alexnet
from utils import cars_downsample

batch = 10

# IF RUNNING WITHOUT A GPU, REMOVE ALL .cuda() FUNCTION CALLS BELOW
class AlexNet:
    def __init__(self):
        # Loads a pre-trained alexnet model.
        model = alexnet(pretrained=True)
        # returns result for the first convolutional layer (after pooling)
        self.c1 = Sequential(*list(model.features.children())[:3]).cuda()
        # returns result for the last convolutional layer (after pooling)
        self.c5 = Sequential(*list(model.features.children())[3:]).cuda()
        # returns result on the FullyConnected layer, just before the output layer of AlexNet
        self.fc2 = Sequential(*list(model.classifier.children())[:-1]).cuda()

    def forward(self, x):
        c1_activations = self.c1(x)  # 1x64x27x27
        c5_activations = self.c5(c1_activations)  # 1x256x13x13
        fc2_activations = self.fc2(c5_activations.view(batch, 9216))  # 4096

        # when we call forward, we learn the features and return them here
        return c1_activations, c5_activations, fc2_activations


if __name__ == "__main__":

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    preprocess_img = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    alex_modified = AlexNet()

    # check if the the image directory exists
    if not os.path.isdir('cars_train') or not os.path.isdir('cars_test'):
        raise FileNotFoundError('Make sure train images are in ./cars_train/[].jpg and test images are in '
                                './cars_test/[].jpg')

    # downsampling on training cars dataset
    labels_path = 'cars_train_annos.mat'
    n = 25 # 25 classes * 40 samples = 1000 total samples
    classes, samples = cars_downsample(labels_path, n)

    X_samples = []
    for sample in samples:
    	X_samples += sample

    X_paths = ['./cars_train/' + car for car in X_samples]
    
    # saving each of the features in a different h5 file
    c1_features = h5py.File('c1_features.h5', 'w')
    c5_features = h5py.File('c5_features.h5', 'w')
    fc2_features = h5py.File('fc2_features.h5', 'w')

    c1_dataset = c1_features.create_dataset('c1', (batch, 64, 27, 27), maxshape=(None, 64, 27, 27))
    c1_labels = c1_features.create_dataset('c1_labels', (batch, ), maxshape=(None, ))
    c5_dataset = c5_features.create_dataset('c5', (batch, 256, 6, 6), maxshape=(None, 256, 6, 6))
    c5_labels = c5_features.create_dataset('c5_labels', (batch, ), maxshape=(None, ))
    fc2_dataset = fc2_features.create_dataset('fc2', (batch, 4096), maxshape=(None, 4096))
    fc2_labels = fc2_features.create_dataset('fc2_labels', (batch, ), maxshape=(None, ))

    # Saving features from Cars dataset
    print("Saving features...")
    new_size = 10
    j = -1
    for i in range(0, len(X_paths), batch):

        if not i % 40:
        	j += 1
        
        print i + 1, 'to', new_size

        if i >= batch:
            c1_dataset.resize(new_size, axis=0)
            c1_labels.resize(new_size, axis=0)
            c5_dataset.resize(new_size, axis=0)
            c5_labels.resize(new_size, axis=0)
            fc2_dataset.resize(new_size, axis=0)
            fc2_labels.resize(new_size, axis=0)

        imgs = []
        labels = []

        for path in X_paths[i:new_size]:
            # savepreprocess_imgs the img in tensor format, ready to be inputted in pytorch alexnet
            imgs.append(preprocess_img(Image.open(path).convert('RGB')).unsqueeze(0))            
            labels.append(classes[j])

        inputImgs = Variable(cat(imgs)).cuda()        

        c1, c5, fc2 = alex_modified.forward(inputImgs)

        labels = np.array(labels)

        c1_dataset[-batch:] = c1.data.cpu().numpy()
        c1_labels[-batch:] = labels
        c5_dataset[-batch:] = c5.data.cpu().numpy()
        c5_labels[-batch:] = labels
        fc2_dataset[-batch:] = fc2.data.cpu().numpy()
        fc2_labels[-batch:] = labels

        new_size += batch

    c1_features.close()
    c5_features.close()
    fc2_features.close()
