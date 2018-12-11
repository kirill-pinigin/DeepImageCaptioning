import torch
import torch.nn as nn
from torchvision import  models
from torchvision.models.squeezenet import Fire
import torch.nn.init as init

LATENT_DIM = int(512)
LATENT_DIM_2 = int(LATENT_DIM // 2) if LATENT_DIM > 2 else 1


class SiLU(torch.nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        out = torch.mul(x, torch.sigmoid(x))
        return out


class EmptyNorm(torch.nn.Module):
    def __init__(self):
        super(EmptyNorm, self).__init__()

    def forward(self, x):
        return x


class FireConvNorm(nn.Module):
    def __init__(self, inplanes=128, squeeze_planes=11,
                 expand1x1_planes=11, expand3x3_planes=11, activation = nn.ReLU(), type_norm = 'batch'):
        super(FireConvNorm, self).__init__()
        self.outplanes = int(expand1x1_planes + expand3x3_planes)
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.activation = activation

        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.type_norm = type_norm
        if self.type_norm == 'instance':
            self.norm_sq = nn.InstanceNorm2d(squeeze_planes)
            self.norm1x1 = nn.InstanceNorm2d(expand1x1_planes)
            self.norm3x3 = nn.InstanceNorm2d(expand3x3_planes)
        elif self.type_norm == 'batch':
            self.norm_sq = nn.BatchNorm2d(squeeze_planes)
            self.norm1x1 = nn.BatchNorm2d(expand1x1_planes)
            self.norm3x3 = nn.BatchNorm2d(expand3x3_planes)
        else:
            self.norm_sq = EmptyNorm()
            self.norm1x1 = EmptyNorm()
            self.norm3x3 = EmptyNorm()

    def ConfigureNorm(self):
        if self.type_norm == 'instance':
            self.norm_sq = nn.InstanceNorm2d(self.squeeze.out_channels)
            self.norm1x1 = nn.InstanceNorm2d(self.expand1x1.out_channels)
            self.norm3x3 = nn.InstanceNorm2d(self.expand3x3.out_channels)
        elif self.type_norm == 'batch':
            self.norm_sq = nn.BatchNorm2d(self.squeeze.out_channels)
            self.norm1x1 = nn.BatchNorm2d(self.expand3x3.out_channels)
            self.norm3x3 = nn.BatchNorm2d(self.expand3x3.out_channels)
        else:
            self.norm_sq = EmptyNorm()
            self.norm1x1 = EmptyNorm()
            self.norm3x3 = EmptyNorm()

    def forward(self, x):
        x = self.activation(self.norm_sq(self.squeeze(x)))
        return torch.cat([
            self.activation(self.norm1x1(self.expand1x1(x))),
            self.activation(self.norm3x3(self.expand3x3(x)))], 1)


class SqueezeSimpleRecognitron(nn.Module):
    def __init__(self, channels= 3, dimension = 35, activation = nn.ReLU(), type_norm = 'batch', pretrained = True):
        super(SqueezeSimpleRecognitron, self).__init__()
        self.activation = activation
        self.dimension = dimension
        if type_norm == 'instance':
            first_norm_layer = nn.InstanceNorm2d(96)
            final_norm_layer = nn.InstanceNorm2d(LATENT_DIM)
        elif type_norm == 'batch':
            first_norm_layer = nn.BatchNorm2d(96)
            final_norm_layer = nn.BatchNorm2d(LATENT_DIM)
        else:
            first_norm_layer = EmptyNorm()
            final_norm_layer = EmptyNorm()

        self.conv1 = nn.Conv2d(channels, 96, kernel_size=7, stride=2)
        self.norm1 = first_norm_layer
        self.downsample1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire1 = FireConvNorm(96, 16, 64, 64, activation=activation, type_norm=type_norm)
        self.fire2 = FireConvNorm(128, 16, 64, 64, activation=activation, type_norm=type_norm)
        self.fire3 = FireConvNorm(128, 32, 128, 128, activation=activation, type_norm=type_norm)
        self.downsample2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire4 = FireConvNorm(256, 32, 128, 128, activation=activation, type_norm=type_norm)
        self.fire5 = FireConvNorm(256, 48, 192, 192, activation=activation, type_norm=type_norm)
        self.fire6 = FireConvNorm(384, 48, 192, 192, activation=activation, type_norm=type_norm)
        self.fire7 = FireConvNorm(384, 64, 256, 256, activation=activation, type_norm=type_norm)
        self.downsample3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire8 = FireConvNorm(512, 64, 256, 256, activation=activation, type_norm=type_norm)
        if pretrained:
            model = models.squeezenet1_0(pretrained=True).features
            if channels == 3:
                self.conv1 = model[0]

            self.fire1.squeeze = model[3].squeeze
            self.fire1.expand1x1 = model[3].expand1x1
            self.fire1.expand3x3 = model[3].expand3x3
            self.fire1.ConfigureNorm()

            self.fire2.squeeze = model[4].squeeze
            self.fire2.expand1x1 = model[4].expand1x1
            self.fire2.expand3x3 = model[4].expand3x3
            self.fire2.ConfigureNorm()

            self.fire3.squeeze = model[5].squeeze
            self.fire3.expand1x1 = model[5].expand1x1
            self.fire3.expand3x3 = model[5].expand3x3
            self.fire3.ConfigureNorm()

            self.fire4.squeeze = model[7].squeeze
            self.fire4.expand1x1 = model[7].expand1x1
            self.fire4.expand3x3 = model[7].expand3x3
            self.fire4.ConfigureNorm()

            self.fire5.squeeze = model[8].squeeze
            self.fire5.expand1x1 = model[8].expand1x1
            self.fire5.expand3x3 = model[8].expand3x3
            self.fire5.ConfigureNorm()

            self.fire6.squeeze = model[9].squeeze
            self.fire6.expand1x1 = model[9].expand1x1
            self.fire6.expand3x3 = model[9].expand3x3
            self.fire6.ConfigureNorm()

            self.fire7.squeeze = model[10].squeeze
            self.fire7.expand1x1 = model[10].expand1x1
            self.fire7.expand3x3 = model[10].expand3x3
            self.fire7.ConfigureNorm()

            self.fire8.squeeze = model[12].squeeze
            self.fire8.expand1x1 = model[12].expand1x1
            self.fire8.expand3x3 = model[12].expand3x3
            self.fire8.ConfigureNorm()

        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_uniform(m.weight)
                    if m.bias is not None:
                        init.constant(m.bias, 0)

        final_conv = nn.Conv2d(512, LATENT_DIM, kernel_size=1)
        init.normal(final_conv.weight, mean=0.0, std=0.01)
        init.constant(final_conv.bias, 0)

        self.features = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            final_norm_layer,
            activation,
            nn.AvgPool2d(12, stride=1),
        )

        self.recognitron = nn.Sequential(
            nn.Linear(LATENT_DIM , LATENT_DIM_2),
            activation,
            nn.Dropout(p=0.5),
            nn.Linear(LATENT_DIM_2, dimension),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.downsample1(x)
        x = self.fire1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.downsample2(x)
        x = self.fire4(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.downsample3(x)
        x = self.fire8(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.recognitron(x)
        return x


class SqueezeResidualRecognitron(nn.Module):
    def __init__(self, channels= 3, dimension = 35, activation = nn.ReLU(), type_norm = 'batch', pretrained = True):
        super(SqueezeResidualRecognitron, self).__init__()
        self.activation = activation
        self.dimension = dimension
        if type_norm == 'instance':
            first_norm_layer = nn.InstanceNorm2d(96)
            final_norm_layer = nn.InstanceNorm2d(LATENT_DIM)
        elif type_norm == 'batch':
            first_norm_layer = nn.BatchNorm2d(96)
            final_norm_layer = nn.BatchNorm2d(LATENT_DIM)
        else:
            first_norm_layer = EmptyNorm()
            final_norm_layer = EmptyNorm()

        self.conv1 = nn.Conv2d(channels, 96, kernel_size=7, stride=2)
        self.norm1 = first_norm_layer
        self.downsample1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire1 = FireConvNorm(96, 16, 64, 64, activation=activation, type_norm=type_norm)
        self.fire2 = FireConvNorm(128, 16, 64, 64, activation=activation, type_norm=type_norm)
        self.fire3 = FireConvNorm(128, 32, 128, 128, activation=activation, type_norm=type_norm)
        self.downsample2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire4 = FireConvNorm(256, 32, 128, 128, activation=activation, type_norm=type_norm)
        self.fire5 = FireConvNorm(256, 48, 192, 192, activation=activation, type_norm=type_norm)
        self.fire6 = FireConvNorm(384, 48, 192, 192, activation=activation, type_norm=type_norm)
        self.fire7 = FireConvNorm(384, 64, 256, 256, activation=activation, type_norm=type_norm)
        self.downsample3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire8 = FireConvNorm(512, 64, 256, 256, activation=activation, type_norm=type_norm)
        if pretrained:
            model = models.squeezenet1_0(pretrained=True).features
            if channels == 3:
                self.conv1 = model[0]

            self.fire1.squeeze = model[3].squeeze
            self.fire1.expand1x1 = model[3].expand1x1
            self.fire1.expand3x3 = model[3].expand3x3
            self.fire1.ConfigureNorm()

            self.fire2.squeeze = model[4].squeeze
            self.fire2.expand1x1 = model[4].expand1x1
            self.fire2.expand3x3 = model[4].expand3x3
            self.fire2.ConfigureNorm()

            self.fire3.squeeze = model[5].squeeze
            self.fire3.expand1x1 = model[5].expand1x1
            self.fire3.expand3x3 = model[5].expand3x3
            self.fire3.ConfigureNorm()

            self.fire4.squeeze = model[7].squeeze
            self.fire4.expand1x1 = model[7].expand1x1
            self.fire4.expand3x3 = model[7].expand3x3
            self.fire4.ConfigureNorm()

            self.fire5.squeeze = model[8].squeeze
            self.fire5.expand1x1 = model[8].expand1x1
            self.fire5.expand3x3 = model[8].expand3x3
            self.fire5.ConfigureNorm()

            self.fire6.squeeze = model[9].squeeze
            self.fire6.expand1x1 = model[9].expand1x1
            self.fire6.expand3x3 = model[9].expand3x3
            self.fire6.ConfigureNorm()

            self.fire7.squeeze = model[10].squeeze
            self.fire7.expand1x1 = model[10].expand1x1
            self.fire7.expand3x3 = model[10].expand3x3
            self.fire7.ConfigureNorm()

            self.fire8.squeeze = model[12].squeeze
            self.fire8.expand1x1 = model[12].expand1x1
            self.fire8.expand3x3 = model[12].expand3x3
            self.fire8.ConfigureNorm()

        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_uniform(m.weight)
                    if m.bias is not None:
                        init.constant(m.bias, 0)

        final_conv = nn.Conv2d(512, LATENT_DIM, kernel_size=1)
        init.normal(final_conv.weight, mean=0.0, std=0.01)
        init.constant(final_conv.bias, 0)

        self.features = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            final_norm_layer,
            activation,
            nn.AvgPool2d(12, stride=1),
        )

        self.recognitron = nn.Sequential(
            nn.Linear(LATENT_DIM, LATENT_DIM_2),
            activation,
            nn.Dropout(p=0.5),
            nn.Linear(LATENT_DIM_2, dimension),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.downsample1(x)
        f1 = self.fire1(x)

        x = self.fire2(f1)
        x = torch.add(x,f1)
        x = self.fire3(x)
        d2 = self.downsample2(x)
        x = self.fire4(d2)
        x = torch.add(x, d2)
        f5 = self.fire5(x)
        x = self.fire6(f5)
        x = torch.add(x, f5)
        x = self.fire7(x)
        d3 = self.downsample3(x)
        x = self.fire8(d3)
        x = torch.add(x, d3)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.recognitron(x)
        return x


class SqueezeShuntRecognitron(nn.Module):
    def __init__(self, channels= 3, dimension = 35, activation = nn.ReLU(),  type_norm = 'batch', pretrained = False):
        super(SqueezeShuntRecognitron, self).__init__()
        self.activation = activation
        self.dimension = dimension
        if type_norm == 'instance':
            first_norm_layer = nn.InstanceNorm2d(96)
            final_norm_layer = nn.InstanceNorm2d(LATENT_DIM)
        elif type_norm == 'batch':
            first_norm_layer = nn.BatchNorm2d(96)
            final_norm_layer = nn.BatchNorm2d(LATENT_DIM)
        else:
            first_norm_layer = EmptyNorm()
            final_norm_layer = EmptyNorm()

        self.conv1 = nn.Conv2d(channels, 96, kernel_size=7, stride=2)
        self.norm1 = first_norm_layer
        self.downsample1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.shunt1 = nn.Sequential(nn.ReLU(), nn.Conv2d(96,128, kernel_size=1), nn.Sigmoid())
        self.fire1 = FireConvNorm(96, 16, 64, 64, activation=activation, type_norm=type_norm)
        self.fire2 = FireConvNorm(128, 16, 64, 64, activation=activation, type_norm=type_norm)
        self.shunt2 = nn.Sequential(nn.ReLU(), nn.Conv2d(128, 256, kernel_size=1), nn.Sigmoid())
        self.fire3 = FireConvNorm(128, 32, 128, 128, activation=activation, type_norm=type_norm)
        self.downsample2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire4 = FireConvNorm(256, 32, 128, 128, activation=activation, type_norm=type_norm)
        self.fire5 = FireConvNorm(256, 48, 192, 192, activation=activation, type_norm=type_norm)
        self.shunt3 = nn.Sequential(nn.ReLU(), nn.Conv2d(256, 384, kernel_size=1), nn.Sigmoid())
        self.fire6 = FireConvNorm(384, 48, 192, 192, activation=activation, type_norm=type_norm)
        self.fire7 = FireConvNorm(384, 64, 256, 256, activation=activation, type_norm=type_norm)
        self.shunt4 = nn.Sequential(nn.ReLU(), nn.Conv2d(384, 512, kernel_size=1), nn.Sigmoid())
        self.downsample3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire8 = FireConvNorm(512, 64, 256, 256, activation=activation, type_norm=type_norm)
        if pretrained:
            model = models.squeezenet1_0(pretrained=True).features
            if channels == 1:
                conv = nn.Conv2d(1, channels, kernel_size=7, stride=2)
                weight = torch.FloatTensor(96, 1, 7, 7)
                parameters = list(model.parameters())
                for i in range(channels):
                    weight[i, :, :, :] = parameters[0].data[i].mean(0)
                    bias = parameters[1].data
                conv.weight.data.copy_(weight)
                conv.bias.data.copy_(bias)
                self.conv1 = conv
            else:
                self.conv1 = model[0]

            self.fire1.squeeze = model[3].squeeze
            self.fire1.expand1x1 = model[3].expand1x1
            self.fire1.expand3x3 = model[3].expand3x3
            self.fire1.ConfigureNorm()

            self.fire2.squeeze = model[4].squeeze
            self.fire2.expand1x1 = model[4].expand1x1
            self.fire2.expand3x3 = model[4].expand3x3
            self.fire2.ConfigureNorm()

            self.fire3.squeeze = model[5].squeeze
            self.fire3.expand1x1 = model[5].expand1x1
            self.fire3.expand3x3 = model[5].expand3x3
            self.fire3.ConfigureNorm()

            self.fire4.squeeze = model[7].squeeze
            self.fire4.expand1x1 = model[7].expand1x1
            self.fire4.expand3x3 = model[7].expand3x3
            self.fire4.ConfigureNorm()

            self.fire5.squeeze = model[8].squeeze
            self.fire5.expand1x1 = model[8].expand1x1
            self.fire5.expand3x3 = model[8].expand3x3
            self.fire5.ConfigureNorm()

            self.fire6.squeeze = model[9].squeeze
            self.fire6.expand1x1 = model[9].expand1x1
            self.fire6.expand3x3 = model[9].expand3x3
            self.fire6.ConfigureNorm()

            self.fire7.squeeze = model[10].squeeze
            self.fire7.expand1x1 = model[10].expand1x1
            self.fire7.expand3x3 = model[10].expand3x3
            self.fire7.ConfigureNorm()

            self.fire8.squeeze = model[12].squeeze
            self.fire8.expand1x1 = model[12].expand1x1
            self.fire8.expand3x3 = model[12].expand3x3
            self.fire8.ConfigureNorm()

        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_uniform(m.weight)
                    if m.bias is not None:
                        init.constant(m.bias, 0)

        final_conv = nn.Conv2d(512, LATENT_DIM, kernel_size=1)
        init.normal(final_conv.weight, mean=0.0, std=0.01)
        init.constant(final_conv.bias, 0)

        self.features = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            final_norm_layer,
            activation,
            nn.AvgPool2d(12, stride=1),
        )

        self.recognitron = nn.Sequential(
            nn.Linear(LATENT_DIM, LATENT_DIM_2),
            activation,
            nn.Dropout(p=0.5),
            nn.Linear(LATENT_DIM_2, dimension),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        d1 = self.downsample1(x)
        f1 = self.fire1(d1)
        s1 = self.shunt1(d1)
        x = torch.add(f1, s1)
        x = self.fire2(x)
        x = torch.add(x,f1)
        s2 = self.shunt2(x)
        x = self.fire3(x)
        x = torch.add(x, s2)
        d2 = self.downsample2(x)
        x = self.fire4(d2)
        x = torch.add(x, d2)
        s3 = self.shunt3(x)
        f5 = self.fire5(x)
        x = torch.add(f5, s3)
        x = self.fire6(x)
        x = torch.add(x, f5)
        s4 = self.shunt4(x)
        x = self.fire7(x)
        x = torch.add(x, s4)
        d3 = self.downsample3(x)
        x = self.fire8(d3)
        x = torch.add(x, d3)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.recognitron(x)
        return x
