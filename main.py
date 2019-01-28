import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from  CSVDataset import  CSVDataset
from PIL import Image
from MultiRecognition import MultiRecognition , MultiLabelLoss, IMAGE_SIZE, CHANNELS, DIMENSION
from NeuralModels import SILU, Perceptron
from ResidualRecognitron import  ResidualRecognitron
from SqueezeRecognitrons import  SqueezeResidualRecognitron
from MobileRecognitron import MobileRecognitron

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',          type = str,   default='/home/user/CocoDatasetTags/', help='path to dataset')
parser.add_argument('--result_dir',        type = str,   default='./RESULTS/', help='path to result')
parser.add_argument('--recognitron',       type = str,   default='MobileRecognitron', help='type of image generator')
parser.add_argument('--activation',        type = str,   default='ReLU', help='type of activation')
parser.add_argument('--criterion',         type = str,   default='BCE', help='type of criterion')
parser.add_argument('--optimizer',         type = str,   default='Adam', help='type of optimizer')
parser.add_argument('--type_norm',         type = str,   default='batch', help='type of optimizer')
parser.add_argument('--lr',                type = float, default=1e-4)
parser.add_argument('--weight_decay',      type = float, default=0)
parser.add_argument('--split',             type = float, default=0.0)
parser.add_argument('--batch_size',        type = int,   default=32)
parser.add_argument('--epochs',            type = int,   default=64)
parser.add_argument('--pretrained',        type = bool,  default=True)
parser.add_argument('--transfer_learning', type = bool,  default=False)
parser.add_argument('--fine_tuning',       type = bool,  default=False)
parser.add_argument('--resume_train',      type = bool,  default=False)

args = parser.parse_args()

recognitron_types = {
                        'ResidualRecognitron'        : ResidualRecognitron,
                        'MobileRecognitron'          : MobileRecognitron,
                        'SqueezeResidualRecognitron' : SqueezeResidualRecognitron,
                    }

activation_types =  {
                        'ReLU'     : nn.ReLU(),
                        'LeakyReLU': nn.LeakyReLU(),
                        'PReLU'    : nn.PReLU(),
                        'ELU'      : nn.ELU(),
                        'SELU'     : nn.SELU(),
                        'SILU'     : SILU()
                    }

criterion_types =   {
                        'MSE'            : nn.MSELoss(),
                        'L1'             : nn.L1Loss(),
                        'BCE'            : nn.BCELoss(),
                        'MultiLabelLoss' : MultiLabelLoss()
                    }

optimizer_types =   {
                        'Adam'   : optim.Adam,
                        'RMSprop': optim.RMSprop,
                        'SGD'    : optim.SGD
                    }

model = (recognitron_types[args.recognitron] if args.recognitron in recognitron_types else recognitron_types['ResidualRecognitron'])

function = (activation_types[args.activation] if args.activation in activation_types else activation_types['ReLU'])

recognitron = model(dimension=DIMENSION , channels=CHANNELS, activation=function,
                    pretrained=args.pretrained + args.transfer_learning + args.fine_tuning)

optimizer =(optimizer_types[args.optimizer] if args.optimizer in optimizer_types else optimizer_types['Adam'])(recognitron.parameters(), lr = args.lr, weight_decay = args.weight_decay)

criterion = (criterion_types[args.criterion] if args.criterion in criterion_types else criterion_types['MSE'])

train_transforms_list = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=(-30, 30), scale=(0.8, 1.2), resample=Image.BICUBIC),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.Resize((int(IMAGE_SIZE * 1.055), int(IMAGE_SIZE * 1.055)), interpolation=3),
        transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        ]

val_transforms_list = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=3),
        transforms.ToTensor(),
        ]

data_transforms = {
    'train':    transforms.Compose(train_transforms_list ),
    'val':      transforms.Compose(val_transforms_list),
}

print(data_transforms)

shuffle_list = { 'train' : True, 'val' : False}

image_datasets = {x: CSVDataset(os.path.join(args.data_dir, x), os.path.join(args.data_dir, x+'_tags.csv'), CHANNELS,
                                          data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,shuffle=shuffle_list[x], num_workers=4)  for x in ['train', 'val']}

testloader =  torch.utils.data.DataLoader(image_datasets['val'], batch_size=1, shuffle=False, num_workers=4)

framework = MultiRecognition(recognitron = recognitron, criterion = criterion, optimizer = optimizer, directory = args.result_dir)

if args.transfer_learning:
    framework.recognitron.freeze()

framework.fit(dataloaders = dataloaders,num_epochs=args.epochs / 2 if args.transfer_learning else args.epochs, resume_train = args.resume_train)

if args.fine_tuning:
    framework.recognitron.unfreeze()
    framework.optimizer = (optimizer_types[args.optimizer] if args.optimizer in optimizer_types else optimizer_types['Adam'])(framework.recognitron.parameters(), lr = args.lr / 2, weight_decay = args.weight_decay)
    framework.fit(dataloaders = dataloaders, num_epochs=args.epochs * 2)

framework.evaluate(testloader)
