import math
import torch
import torch.nn as nn

import torch.nn.functional as F
from torchvision import models


class SiLU(torch.nn.Module):
    def __init__(self, inplace=False):
        super(SiLU, self).__init__()

    def forward(self, x):
        out = torch.mul(x, F.sigmoid(x))
        return out


class ResidualRecognitron(nn.Module):
    def __init__(self, channels = 1, dimension=11, activation = SiLU(), pretrained = True):
        super(ResidualRecognitron, self).__init__()
        self.activation = activation
        self.model = models.resnet18(pretrained=pretrained)
        if pretrained == True and channels == 1:
            conv = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            weight = torch.FloatTensor(64, 1, 7, 7)
            parameters = list(self.model.parameters())
            for i in range(64):
                weight[i, :, :, :] = parameters[0].data[i].mean(0)
            conv.weight.data.copy_(weight)
            self.model.conv1 = conv
        self.model.relu = activation
        self.model.avgpool = nn.AvgPool2d(8, stride=1)
        num_ftrs = self.model.fc.in_features
        print ('! fc.in_features ', num_ftrs)
        sub_dimension = int(math.sqrt(num_ftrs))
        av_dimension = int( float(dimension + num_ftrs) /2.0)
        reduce_number = sub_dimension if sub_dimension > av_dimension else av_dimension
        self.model.fc = nn.Sequential(
            #activation,
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, dimension),
            #activation,
           # nn.Dropout(p=0.5),
           # nn.Linear(reduce_number, dimension),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
