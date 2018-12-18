import time
import sys
import os
import torch
from torch.autograd import Variable
import torchvision
import numpy as np

IMAGE_SIZE = 224
CHANNELS = 3

LR_THRESHOLD = 1e-7
TRYING_LR = 5
DEGRADATION_TOLERANCY = 5
ACCURACY_TRESHOLD = float(0.0625)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


class MultiRecognition(object):
    def __init__(self, recognitron,  criterion, optimizer, dataloaders, directory):
        self.recognitron = recognitron
        self.criterion = criterion
        self.optimizer = optimizer
        self.use_gpu = torch.cuda.is_available()
        self.dataloaders = dataloaders

        config = str(recognitron.__class__.__name__) + '_' + str(recognitron.activation.__class__.__name__) #+ '_' + str(recognitron.norm1.__class__.__name__)
        if str(recognitron.__class__.__name__) == 'ResidualRecognitron':
            config += str(recognitron.model.conv1)
        config += '_' + str(criterion.__class__.__name__)
        config += "_" + str(optimizer.__class__.__name__) #+ "_lr_" + str( optimizer.param_groups[0]['lr'])

        reportPath = os.path.join(directory, config + "/report/")
        flag = os.path.exists(reportPath)
        if flag != True:
            os.makedirs(reportPath)
            print('os.makedirs("reportPath")')

        self.modelPath = os.path.join(directory, config + "/model/")
        flag = os.path.exists(self.modelPath)
        if flag != True:
            os.makedirs(self.modelPath)
            print('os.makedirs("/modelPath/")')

        self.images = os.path.join(directory, config + "/images/")
        flag = os.path.exists(self.images)
        if flag != True:
            os.makedirs(self.images+'/bad/')
            os.makedirs(self.images + '/good/')
            print('os.makedirs("/images/")')

        self.report = open(reportPath  + '/' + config + "_Report.txt", "w")
        _stdout = sys.stdout
        sys.stdout = self.report
        print(config)
        print(recognitron)
        print(criterion)
        self.report.flush()
        sys.stdout = _stdout
        if self.use_gpu :
            self.recognitron = self.recognitron.cuda()

    def __del__(self):
        self.report.close()

    def train(self, num_epochs = 20, resume_train = False):
        if resume_train and os.path.isfile(self.modelPath + 'BestRecognitron.pth'):
            print( "RESUME training load BestRecognitron")
            self.recognitron.load_state_dict(torch.load(self.modelPath + 'BestRecognitron.pth'))

        since = time.time()
        best_loss = 10000.0
        best_acc = 0.0
        counter = 0
        i = int(0)
        degradation = 0
        for epoch in range(num_epochs):
            _stdout = sys.stdout
            sys.stdout = self.report
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            self.report.flush()
            sys.stdout = _stdout
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.recognitron.train(True)
                else:
                    self.recognitron.train(False)

                running_loss = 0.0
                running_corrects = 0
                for data in self.dataloaders[phase]:
                    inputs, targets = data
                    if self.use_gpu:
                        inputs = Variable(inputs.cuda())
                        targets = Variable(targets.cuda())
                    else:
                        inputs, targets = Variable(inputs), Variable(targets)
                    self.optimizer.zero_grad()

                    outputs = self.recognitron(inputs)
                    diff = torch.abs(targets.data - torch.round(outputs.data))
                    loss = self.criterion(outputs, targets)
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += (1.0 - torch.sum(diff)/float(diff.shape[1]*diff.shape[0])) * inputs.size(0)

                epoch_loss = float(running_loss) / float(len(self.dataloaders[phase].dataset))
                epoch_acc = float(running_corrects) / float(len(self.dataloaders[phase].dataset))

                _stdout = sys.stdout
                sys.stdout = self.report
                print('{} Loss: {:.4f} Accuracy {:.4f} '.format(
                    phase, epoch_loss, epoch_acc))
                self.report.flush()

                sys.stdout = _stdout
                print('{} Loss: {:.4f} Accuracy {:.4f} '.format(
                    phase, epoch_loss, epoch_acc))
                self.report.flush()

                if phase == 'val' and epoch_loss < best_loss:
                    counter = 0
                    degradation = 0
                    best_loss = epoch_loss
                    print('curent best_loss ', best_loss)
                    self.save('/BestRecognitron')
                else:
                    counter += 1
                    self.save('/RegualarRecognitron')

            if counter > TRYING_LR * 2:
                for param_group in self.optimizer.param_groups:
                    lr = param_group['lr']
                    if lr >= LR_THRESHOLD:
                        param_group['lr'] = lr * 0.5
                        print('! Decrease LearningRate !', lr)
                counter = 0
                degradation += 1
            if degradation > DEGRADATION_TOLERANCY:
                print('This is the end! Best val best_loss: {:4f}'.format(best_loss))
                return best_loss

        time_elapsed = time.time() - since

        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val best_loss: {:4f}'.format(best_loss))
        return best_loss


    def evaluate(self, test_loader, isSaveImages = True, modelPath=None):
        tags = test_loader.dataset.tags()
        counter = 0
        if modelPath is not None:
            self.recognitron.load_state_dict(torch.load(modelPath))
            print('load recognitron model')
        else:
            self.recognitron.load_state_dict(torch.load(self.modelPath + 'BestRecognitron.pth'))
            print('load BestRecognitron ')
        print(len(test_loader.dataset))
        i = 0
        since = time.time()
        self.recognitron.train(False)
        self.recognitron.eval()
        if self.use_gpu:
            self.recognitron = self.recognitron.cuda()
        running_loss = 0.0
        running_corrects = 0
        for data in test_loader:
            inputs, targets = data

            if self.use_gpu:
                inputs = Variable(inputs.cuda())
                targets = Variable(targets.cuda())
            else:
                inputs, targets = Variable(inputs), Variable(targets)


            outputs = self.recognitron(inputs)
            diff = torch.abs(targets.data - torch.round(outputs.data))
            loss = self.criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (1.0 - torch.sum(diff) / float(diff.shape[1] * diff.shape[0])) * inputs.size(0)

            result = torch.round(outputs).data.cpu().numpy()
            result = np.squeeze(result)
            indexes = np.nonzero(result)
            image = inputs.clone()
            image = image.data.cpu().float()
            counter = counter + 1
            filename = self.images + '/' + str(counter)
            labels = tags[indexes]
            for l in labels:
                filename += '_' + str(l) + '_'
            filename+='.png'
            torchvision.utils.save_image(image, filename)

        epoch_loss = float(running_loss) / float(len(test_loader.dataset))
        epoch_acc = float(running_corrects) / float(len(test_loader.dataset))

        time_elapsed = time.time() - since

        print('Evaluating complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Loss: {:.4f} Accuracy {:.4f} '.format( epoch_loss, epoch_acc))
        #self.report.flush()

    def save(self, model):
        self.recognitron = self.recognitron.cpu()
        x = Variable(torch.zeros(1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
        torch_out = torch.onnx._export(self.recognitron, x,self.modelPath + '/' + model + ".onnx", export_params=True)
        torch.save(self.recognitron.state_dict(), self.modelPath + '/' + model + ".pth")

        if self.use_gpu:
            self.recognitron = self.recognitron.cuda()


class MultiLabelLoss(torch.nn.Module):
    def __init__(self, channels =3):
        super(MultiLabelLoss, self).__init__()
        self.criterion = torch.nn.BCELoss()
        self.penalty = torch.nn.L1Loss()
        self.loss = None

    def forward(self, input, target):
        p = input.data.cpu().numpy()
        t = target.data.cpu().numpy()
        #positive_input = Variable(torch.from_numpy(p[np.nonzero(t)]))
        #positive_target = Variable(torch.from_numpy(t[np.nonzero(t)]))
        negative_input = Variable(torch.from_numpy(p[np.where(t == 0)]))
        negative_target = Variable(torch.from_numpy(t[np.where(t == 0)]))
        if torch.cuda.is_available():
            #positive_input  = positive_input .cuda()
            #positive_target = positive_target.cuda()
            negative_input  = negative_input .cuda()
            negative_target = negative_target.cuda()

        bce_loss = self.criterion(input, target)
        #false_positive_penalty = self.penalty(positive_input, positive_target)
        false_negative_penalty= self.penalty(negative_input, negative_target)
        self.loss =  bce_loss + false_negative_penalty
        return self.loss

    def backward(self, retain_variables=True):
        return self.loss.backward(retain_variables=retain_variables)