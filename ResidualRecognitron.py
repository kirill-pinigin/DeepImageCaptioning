import torch
import torch.nn as nn

from torchvision import models


class SiLU(torch.nn.Module):
    def __init__(self, inplace=False):
        super(SiLU, self).__init__()

    def forward(self, x):
        out = torch.mul(x, torch.sigmoid(x))
        return out

class EmptyNorm(torch.nn.Module):
    def __init__(self):
        super(EmptyNorm, self).__init__()

    def forward(self, x):
        return x

class ResidualRecognitron(nn.Module):
    def __init__(self, channels = 1, dimension=11, activation = SiLU(), type_norm='batch', pretrained = True):
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

        if type_norm == 'instance':
            self.model.bn1 = nn.InstanceNorm1d(64)

        self.model.relu = activation
        self.model.avgpool = nn.AvgPool2d(7)
        num_ftrs = self.model.fc.in_features
        reduce_number = int ((num_ftrs + dimension)/2.0)
        sub_dimension = reduce_number if reduce_number < dimension else (reduce_number + dimension)

        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, sub_dimension),
            activation,
            nn.Dropout(p=0.5),
            nn.Linear(sub_dimension, dimension),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True