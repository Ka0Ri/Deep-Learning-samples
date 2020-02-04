import torch
import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger


import argparse
import numpy as np
import os

data_path = os.path.dirname(os.getcwd()) + "/data/"


class Options(object):

    def __init__(self):
        """
        setting parameters
        """
        self.parser = argparse.ArgumentParser(description='Neural Network with Visdom')
        self.parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
        self.parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        self.parser.add_argument('--weight_decay', default=5e-4, type=float)
        self.parser.add_argument('--n_epoches', default=10, type=int, help='Number of training epoches')
        self.parser.add_argument('--n_batches', default=256, type=int, help='batch size')
        self.parser.add_argument('--device', default='cuda', type=str, help='running environment')
        self.parser.add_argument('--n_classes', default=10, type=int, help='number of classes')
        self.parser.add_argument('--log_file', default='log', type=str, help='name of log file')

        self.opt = self.parser.parse_args()
    
    def setup_config(self):
        if torch.cuda.is_available():
            #torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.opt.device = 'cuda:0'
        else:
            #torch.set_default_tensor_type('torch.FloatTensor')
            self.opt.device = 'cpu'

option = Options()
option.setup_config()
args = option.opt


class Customized_Logger():
    """Write log to file and save the model"""
    def __init__(self, file_name):

        self.file_name = file_name
        self.best_acc = 0
        self.best_epoch = -1
        with open('logs/' + self.file_name + '.txt', 'w+') as log_file:
            log_file.write("--SETTINGS--\n")
            for arg in vars(args):
                log_file.write('%s: %s\n'%(arg, getattr(args, arg)))
            log_file.write("--WRITE LOG--\n")
            log_file.write("train_acc\tval_acc\train_loss\tval_loss\n\n")

    def update(self, train_acc, train_err, val_acc, val_err, epoch, model, save_best = True):
        if(save_best == True):
            if(val_acc > self.best_acc):
                self.best_acc = val_acc
                self.best_epoch = epoch
                torch.save(model.state_dict(), 'logs/' + self.file_name + '.pt')
        else:
            torch.save(model.state_dict(), 'logs/' + self.file_name + '.pt')

        with open('logs/' + self.file_name + '.txt', 'r+') as log_file:
            lines = log_file.readlines()
            lines[-1] = '%.4f\t%.4f\t%.4f\t%.4f\n'%(train_acc, val_acc, train_err, val_err)
            log_file.seek(0)
            log_file.writelines(lines)
            log_file.write('Best accuracy %.4f at epoch %d\n'%(self.best_acc, self.best_epoch))

class Visualier():
    """Visulization, plot the logs during training process"""
    def __init__(self, num_classes=10):

        port = 8097
        self.loss_logger = VisdomPlotLogger('line', port=port, win = "Loss", opts={'title': 'Loss Logger'})
        self.acc_logger = VisdomPlotLogger('line', port=port, win = "acc", opts={'title': 'Accuracy Logger'})
        self.confusion_logger = VisdomLogger('heatmap', port=port, win="confusion", opts={'title': 'Confusion matrix',
                                                                'columnnames': list(range(num_classes)),
                                                                'rownames': list(range(num_classes))})
    
    def plot(self, train_acc, train_err, val_acc, val_err, confusion, epoch):
        self.loss_logger.log(epoch, train_err, name="train")
        self.acc_logger.log(epoch, train_acc, name="train")
        self.loss_logger.log(epoch, val_err, name="val")
        self.acc_logger.log(epoch, val_acc, name="val")
        self.confusion_logger.log(confusion)

        print("epoch: [%d/%d]"%(epoch, args.n_epoches))
        print('Training loss: %.4f, accuracy: %.2f%%' % (train_err, train_acc))
        print('Validation loss: %.4f, accuracy: %.2f%%' % (val_err, val_acc))