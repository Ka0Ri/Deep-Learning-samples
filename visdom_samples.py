
""" Run MNIST example and log to visdom
    Notes:
        - Visdom must be installed (pip works)
        - the Visdom server must be running at start!
    Example:
        $ python -m visdom.server -port 8097 &
        $ python mnist_with_visdom.py
"""
from tqdm import tqdm
import torch
import torch.optim
import torch.nn as nn
import torchnet as tnt
from torch.autograd import Variable
import torch.nn.functional as F
from torchnet.engine import Engine
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from ults import *

import numpy as np
import os

data_path = os.path.dirname(os.getcwd()) + "/data/"


option = Options()
option.setup_config()
args = option.opt


class FashionMnistread(TensorDataset):
    """Customized dataset loader"""
    def __init__(self, mode, transform):
        dataset = FashionMNIST(root=data_path, download=True, train=mode)
        data = getattr(dataset, 'train_data' if mode else 'test_data')
        labels = getattr(dataset, 'train_labels' if mode else 'test_labels')
        self.transform = transform
        self.input_images = np.array(data).astype(np.float)
        self.input_labels = np.array(labels).astype(np.long)

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx]
        labels = self.input_labels[idx]
        if self.transform is not None:
            images = self.transform(images)
        return images, labels

class My_Model(nn.Module):
    """Model Definition"""
    def __init__(self, num_of_class):
        super(My_Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_of_class)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        return out


def main():

    ###Initialization
    device = torch.device(args.device)

    My_transform = transforms.Compose([
        transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
        ])

    Train_data = FashionMnistread(True, transform=My_transform)
    Test_data = FashionMnistread(False, transform=My_transform)

    Train_dataloader = DataLoader(dataset=Train_data, batch_size = args.n_batches, shuffle=False)
    Test_dataloader = DataLoader(dataset=Test_data, batch_size = args.n_batches, shuffle=False)

    def get_iterator(mode):
        if mode is True:
            return Train_dataloader
        elif mode is False:
            return Test_dataloader

    
    from torchsummary import summary
    _model = My_Model(num_of_class = args.n_classes)
    _model.to(device)
    summary(_model, input_size=(1, 28, 28))

    optimizer = torch.optim.SGD(_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    classerr = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(args.n_classes, normalized=True)

    plotLogger = Visualier(num_classes = args.n_classes)
    writelogger = Customized_Logger(file_name=args.log_file)
    ###End Initialization

    def h(sample):
        data, classes, training = sample
       
        _model.train() if training else _model.eval()
            
        labels = torch.LongTensor(classes).to(device)
        data = data.to(device).float()
        
        f_class = _model(data)
        loss = criterion(f_class, labels)
        
        p_class = F.softmax(f_class, dim=1)
        return loss, p_class
            

    def reset_meters():
        classerr.reset()
        meter_loss.reset()
        confusion_meter.reset()

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        classerr.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].item())

    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(state):
        
        train_acc =  classerr.value()[0]
        train_err = meter_loss.value()[0]

        # do validation at the end of each epoch
        reset_meters()
        engine.test(h, get_iterator(False))
   
        val_acc =  classerr.value()[0]
        val_err = meter_loss.value()[0]
        plotLogger.plot(train_acc=train_acc, train_err=train_err, val_acc=val_acc, val_err=val_err, 
                        confusion = confusion_meter.value(),
                        epoch =state['epoch'])
        writelogger.update(train_acc=train_acc, train_err=train_err, val_acc=val_acc, val_err=val_err, epoch=state['epoch'], model=_model)
        

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.train(h, get_iterator(True), maxepoch=args.n_epoches, optimizer=optimizer)


if __name__ == '__main__':
    main()