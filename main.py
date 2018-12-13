import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from  CSVDataset import  CSVDataset , make_dataloaders

from MultiRecognition import MultiRecognition
from ResidualRecognitron import  ResidualRecognitron, SiLU
from SqueezeRecognitrons import  SqueezeResidualRecognitron

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',          type = str,   default='/home/user/CocoDatasetTags/', help='path to dataset')
parser.add_argument('--result_dir',        type = str,   default='./RESULTS/', help='path to result')
parser.add_argument('--recognitron',       type = str,   default='ResidualRecognitron', help='type of image generator')
parser.add_argument('--activation',        type = str,   default='ReLU', help='type of activation')
parser.add_argument('--criterion',         type = str,   default='BCE', help='type of criterion')
parser.add_argument('--optimizer',         type = str,   default='RMSprop', help='type of optimizer')
parser.add_argument('--lr',                type = float, default='2e-5')
parser.add_argument('--split',             type = float, default='0.0')
parser.add_argument('--dimension',         type = int,   default='35')
parser.add_argument('--channels',          type = int,   default='3')
parser.add_argument('--batch_size',        type = int,   default='32')
parser.add_argument('--epochs',            type = int,   default='201')
parser.add_argument('--augmentation',      type = bool,  default='True', help='type of training')
parser.add_argument('--pretrained',        type = bool,  default='True', help='type of training')
parser.add_argument('--transfer_learning', type = bool,  default='True', help='type of training')
parser.add_argument('--fine_tuning',       type = bool,  default='True', help='type of training')

IMAGE_SIZE = 224

print(torch.__version__)
args = parser.parse_args()

recognitron_types = {   'ResidualRecognitron'                : ResidualRecognitron,
                        'SqueezeResidualRecognitron'         : SqueezeResidualRecognitron,
                    }

activation_types = {'ReLU'     : nn.ReLU(),
                    'LeakyReLU' : nn.LeakyReLU(),
                    'PReLU'    : nn.PReLU(),
                    'ELU'      : nn.ELU(),
                    'SELU'     : nn.SELU(),
                    'SiLU'     : SiLU()
              }

criterion_types = {
                    'MSE' : nn.MSELoss(),
                    'L1'  : nn.L1Loss(),
                    'BCE' : nn.BCELoss(),
                    'MultiLabelSoftMarginLoss' : nn.MultiLabelSoftMarginLoss(),
                    }

optimizer_types = {
                    'Adam'           : optim.Adam,
                    'RMSprop'       : optim.RMSprop,
                    'SGD'           : optim.SGD
                    }

model = (recognitron_types[args.recognitron] if args.recognitron in recognitron_types else recognitron_types['ResidualRecognitron'])
function = (activation_types[args.activation] if args.activation in activation_types else activation_types['ReLU'])

recognitron = model(dimension=args.dimension , channels=args.channels, activation=function, pretrained=args.pretrained + args.transfer_learning + args.fine_tuning)

optimizer =(optimizer_types[args.optimizer] if args.optimizer in optimizer_types else optimizer_types['Adam'])(recognitron.parameters(), lr = args.lr)

criterion = (criterion_types[args.criterion] if args.criterion in criterion_types else criterion_types['MSE'])

train_transforms_list =[
        transforms.RandomHorizontalFlip(),
        transforms.Resize((240, 240), interpolation=3), transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
        #transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=3),
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

image_datasets = {x: CSVDataset(os.path.join(args.data_dir, x), os.path.join(args.data_dir, x+'_tags.csv'), args.channels,
                                          data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,shuffle=shuffle_list[x], num_workers=4)  for x in ['train', 'val']}

framework = MultiRecognition(recognitron = recognitron, criterion = criterion, optimizer = optimizer, dataloaders = dataloaders,  directory = args.result_dir)

if args.transfer_learning:
    framework.recognitron.freeze()

framework.train(num_epochs=args.epochs)

if args.fine_tuning:
    framework.recognitron.unfreeze()
    framework.optimizer = (optimizer_types[args.optimizer] if args.optimizer in optimizer_types else optimizer_types['Adam'])(framework.recognitron.parameters(), lr=args.lr / 2)
    framework.train(num_epochs=args.epochs)

framework.evaluate(dataloaders['val'])
