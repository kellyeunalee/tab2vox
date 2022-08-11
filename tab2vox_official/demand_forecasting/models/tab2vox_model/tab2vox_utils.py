import sys
import numpy as np

import torch

class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = f'{self.name:10s} {self.avg:.3f}'
        return fmtstr
    
class ProgressMeter(object):
    def __init__(self, meters, loader_length, prefix=""):
        self.meters = [AverageMeter(i) for i in meters]
        self.loader_length = loader_length
        self.prefix = prefix
    
    def reset(self):
        for m in self.meters:
            m.reset()
    
    def update(self, values, n=1):
        for m, v in zip(self.meters, values):
            m.update(v, n)
            self.__setattr__(m.name, m.avg)

    def display(self, batch_idx, postfix=""):
        batch_info = f'[{batch_idx+1:03d}/{self.loader_length:03d}]'
        msg = [self.prefix + ' ' + batch_info]
        msg += [str(meter) for meter in self.meters]
        msg = ' | '.join(msg)

        sys.stdout.write('\r')
        sys.stdout.write(msg + postfix)
        sys.stdout.flush()
              

def accuracy(logits, targets):

    logits = torch.round(torch.nn.functional.relu(torch.squeeze(logits), inplace=True))
    targets = torch.round(torch.nn.functional.relu(targets, inplace=True))

    lst = []
    for i in range(targets.size(0)): 
        
        y_hat = logits[i].item() 
        y = targets[i].item()
        try:
            acc_i = min(y_hat, y)/max(y_hat, y)
        except ZeroDivisionError:   
            acc_i = 1.0
        lst.append(acc_i)
    acc = np.mean(lst)
    return acc

def test_accuracy(logits, targets):

    logits = torch.round(torch.nn.functional.relu(logits, inplace=True))
    targets = torch.round(torch.nn.functional.relu(targets, inplace=True))

    y_hat = logits.item() 
    y = targets.item()
    try:
        acc = min(y_hat, y)/max(y_hat, y)
    except ZeroDivisionError:  
        acc = 1.0

    return acc