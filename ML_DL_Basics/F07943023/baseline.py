###################################################################################
## Problem 4(b):                                                                 ##
## You should extract image features using pytorch pretrained alexnet and train  ##
## a KNN classifier to perform face recognition as your baseline in this file.   ##
###################################################################################

import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision.models import alexnet
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import os


def get_dataloader(folder, output_figure, batch_size=32):
    # Data preprocessing
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    train_path, test_path = os.path.join(
        folder, 'train'), os.path.join(folder, 'valid')

    # Get dataset using pytorch functions
    train_set = ImageFolder(train_path, transform=trans)
    test_set = ImageFolder(test_path,  transform=trans)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,  batch_size=batch_size, shuffle=False)
    print('==>>> total training batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))

    # Use pre-trained model
    extractor = alexnet(pretrained=True).features
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        extractor.cuda()
    extractor.eval()

    train_feat = None  # (6987, 256)
    train_label = None
    test_feat = None  # (1526, 256)
    test_label = None
    with torch.no_grad():

        for batch, (x, label) in enumerate(train_loader, 1):

            # Put input tensor to GPU if it's available
            if use_cuda:
                x, label = x.cuda(), label.cuda()
            # Forward input tensor through alexnet
            feat = extractor(x).view(x.size(0), 256, -1)
            feat = torch.mean(feat, 2)
            feat = feat.cpu().numpy()
            label = label.cpu().numpy()

            if train_feat is None:
                train_feat = feat
                train_label = label
            else:
                train_feat = np.append(train_feat, feat, axis=0)
                train_label = np.append(train_label, label, axis=0)

        '''
        first_10_train = train_feat[train_label<10]         #(692,256)
        first_10_train_label = train_label[train_label<10]

        tsne_output = TSNE(n_components=2).fit_transform(first_10_train)   #(120,2)
        plt.scatter(tsne_output[:,0], tsne_output[:,1], cmap = 'tab20', c=first_10_train_label)

        plt.title("TSNE of (1) feature for training data", weight = 'bold').set_fontsize(14)
        plt.xlabel("Dimension 1").set_fontsize(10)
        plt.ylabel("Dimension 2").set_fontsize(10)
        plt.savefig("baseline_tsne_training"+".png")
        '''

        for batch, (x, label) in enumerate(test_loader, 1):

            if use_cuda:
                x, label = x.cuda(), label.cuda()

            feat = extractor(x).view(x.size(0), 256, -1)
            feat = torch.mean(feat, 2)
            feat = feat.cpu().numpy()
            label = label.cpu().numpy()

            if test_feat is None:
                test_feat = feat
                test_label = label
            else:
                test_feat = np.append(test_feat, feat, axis=0)
                test_label = np.append(test_label, label, axis=0)

        first_10_test = test_feat[test_label < 10]  # (145,256)
        first_10_test_label = test_label[test_label < 10]

        tsne_output = TSNE(n_components=2).fit_transform(first_10_test)
        plt.scatter(tsne_output[:, 0], tsne_output[:, 1],
                    cmap='tab20', c=first_10_test_label)

        plt.title("TSNE of (1) feature for validation data",
                  weight='bold').set_fontsize(14)
        plt.xlabel("Dimension 1").set_fontsize(10)
        plt.ylabel("Dimension 2").set_fontsize(10)
        plt.savefig(output_figure)

    # PCA
    pca = PCA(n_components=100)
    pca.fit(train_feat)
    pca_train_feat = pca.transform(train_feat)  # (6987, 100)
    pca_test_feat = pca.transform(test_feat)  # (1526,100)

    return pca_train_feat, train_label, pca_test_feat, test_label


if __name__ == "__main__":
    # TODO

    # Specifiy data folder path and output figure
    folder, output_figure = sys.argv[1], sys.argv[2]

    # Get data loaders of training set and validation set
    pca_train_feat, train_label, pca_test_feat, test_label = get_dataloader(folder, output_figure, batch_size=32)

    # Training KNN classifier
    knn = KNeighborsClassifier(n_neighbors=100)
    knn.fit(pca_train_feat, train_label)

    print('Recognition rate: ' + str(knn.score(pca_test_feat, test_label)))
