from glob import glob
import os

import h5py
from PIL import Image
from torch import cat
from torch.autograd import Variable
from torch.nn import Sequential
from torchvision import transforms
from torchvision.models import alexnet

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

    # pre-processing of the
    preprocessFn = transforms.Compose([transforms.Scale(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                            std=[0.5, 0.5, 0.5])])

    alex_modified = AlexNet()

    # check if the the image directory exists
    if not os.path.isdir('cars_train') or not os.path.isdir('cars_test'):
        raise FileNotFoundError('Make sure train images are in ./cars_train/[].jpg and test images are in '
                                './cars_test/[].jpg')

    train_paths = sorted(glob('cars_train/*.jpg'))  # 8144 814*10, 4
    test_paths = sorted(glob('cars_test/*.jpg'))  # 8041 804*10, 1

    # saving each of the features in a different h5 file
    c1_dataset = h5py.File('c1_features.h5', 'w')
    c5_dataset = h5py.File('c5_features.h5', 'w')
    fc2_dataset = h5py.File('fc2_features.h5', 'w')

    # split them into test/train -> also, specify their sizes
    c1_train_dataset = c1_dataset.create_dataset('c1_train', (batch, 64, 27, 27), maxshape=(None, 64, 27, 27))
    c1_test_dataset = c1_dataset.create_dataset('c1_test', (batch, 64, 27, 27), maxshape=(None, 64, 27, 27))

    c5_train_dataset = c5_dataset.create_dataset('c5_train', (batch, 256, 6, 6), maxshape=(None, 256, 6, 6))
    c5_test_dataset = c5_dataset.create_dataset('c5_test', (batch, 256, 6, 6), maxshape=(None, 256, 6, 6))

    fc2_train_dataset = fc2_dataset.create_dataset('fc2_train', (batch, 4096), maxshape=(None, 4096))
    fc2_test_dataset = fc2_dataset.create_dataset('fc2_test', (batch, 4096), maxshape=(None, 4096))

    # Saving training features from Cars dataset
    print("Saving training features...")
    for i in range(0, len(train_paths), batch):

        if i >= 8140:  # train dataset size
            new_size = i + 4
        else:
            new_size = i + batch

        print(i + 1, 'to', new_size)

        if i >= batch:
            c1_train_dataset.resize(new_size, axis=0)
            c5_train_dataset.resize(new_size, axis=0)
            fc2_train_dataset.resize(new_size, axis=0)

        if i >= 8140:
            batch = 4

        imgs = []

        for path in train_paths[i:new_size]:
            # saves the img in tensor format, ready to be inputted in pytorch alexnet
            imgs.append(preprocessFn(Image.open(path).convert('RGB')).unsqueeze(0))

        inputImgs = Variable(cat(imgs)).cuda()

        c1, c5, fc2 = alex_modified.forward(inputImgs)

        c1_train_dataset[-batch:] = c1.data.cpu().numpy()
        c5_train_dataset[-batch:] = c5.data.cpu().numpy()
        fc2_train_dataset[-batch:] = fc2.data.cpu().numpy()

    batch = 10

    # Saving testing features from Cars dataset
    print("Saving testing features...")
    for i in range(0, len(test_paths), batch):

        if i >= 8040:  # test dataset size
            new_size = i + 1
        else:
            new_size = i + batch

        print(i + 1, 'to', new_size)

        if i >= batch:
            c1_test_dataset.resize(new_size, axis=0)
            c5_test_dataset.resize(new_size, axis=0)
            fc2_test_dataset.resize(new_size, axis=0)

        if i >= 8040:
            batch = 1

        imgs = []

        for path in train_paths[i:new_size]:
            imgs.append(preprocessFn(Image.open(path).convert('RGB')).unsqueeze(0))

        inputImgs = Variable(cat(imgs)).cuda()

        c1, c5, fc2 = alex_modified.forward(inputImgs)

        c1_test_dataset[-batch:] = c1.data.cpu().numpy()
        c5_test_dataset[-batch:] = c5.data.cpu().numpy()
        fc2_test_dataset[-batch:] = fc2.data.cpu().numpy()

    c1_dataset.close()
    c5_dataset.close()
    fc2_dataset.close()
