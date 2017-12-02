import h5py
import numpy as np
import sys

from glob import glob
from PIL import Image
from torch.autograd import Variable
from torch import cat, stack
from torch.nn import Sequential
from torchvision import transforms
from torchvision.models import alexnet

batch = 10


class AlexNet():
    def __init__(self):
        model = alexnet(pretrained=True)
        self.c1 = Sequential(*list(model.features.children())[:3]).cuda()
        self.c5 = Sequential(*list(model.features.children())[3:]).cuda()
        # self.m3 = Sequential(*list(model.features.children())[3:]).cuda()
        self.fc2 = Sequential(*list(model.classifier.children())[:-1]).cuda()

    def forward(self, x):
        c1_activations = self.c1(x)  # 1x64x27x27
        c5_activations = self.c5(c1_activations)  # 1x256x13x13
        # m3_activations = self.m3(c1_activations).view(batch, 9216)  # 9216
        fc2_activations = self.fc2(c5_activations.view(batch, 9216))  # 4096
        return c1_activations, c5_activations, fc2_activations


if __name__ == "__main__":

    preprocessFn = transforms.Compose([transforms.Scale(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

    alex_modified = AlexNet()

    train_paths = sorted(glob('cars_train/*.jpg'))  # 8144 814*10, 4
    test_paths = sorted(glob('cars_test/*.jpg'))  # 8041 804*10, 1

    c1_dataset = h5py.File('c1_features.h5', 'w')
    c5_dataset = h5py.File('c5_features.h5', 'w')
    fc2_dataset = h5py.File('fc2_features.h5', 'w')

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
