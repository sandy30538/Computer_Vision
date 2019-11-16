import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # TODO
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(4, 4))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, x):
        # TODO
        out = self.convnet(x)
        out = out.view(x.size(0), -1)
        out = self.fc(out)

        return out

    def name(self):
        return "ConvNet"

class Fully(nn.Module):
    def __init__(self):
        super(Fully, self).__init__()
        # TODO

        self.fc = nn.Sequential(OrderedDict([
            ('f1', nn.Linear(784, 576)),
            ('relu1', nn.ReLU()),
            ('f2', nn.Linear(576, 144)),
            ('relu2', nn.ReLU()),
            ('f3', nn.Linear(144, 120)),
            ('relu3', nn.ReLU()),
            ('f4', nn.Linear(120, 84)),
            ('relu4', nn.ReLU()),
            ('f5', nn.Linear(84, 10)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, x):
        x = x.view(x.size(0),-1) # flatten input tensor
        # TODO
        out = self.fc(x)
        return out

    def name(self):
        return "Fully"

