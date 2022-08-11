
import time
from time import localtime
from time import strftime
import pandas as pd

import torch
from demand_forecasting.models.tab2vox_model.tab2vox_utils import ProgressMeter, accuracy, test_accuracy

import torch.nn as nn

class Tab2voxTrainer(object):
    def __init__(self, model, criterion, optimizer, scheduler, device, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_epoch, self.best_acc = 0, 0
        self.config = config
        
    def train(self, train_loader, epoch):
        progress = ProgressMeter(["train_loss", "train_acc"], len(train_loader), prefix=f'EPOCH {epoch:03d}')
        self.model.train()
        self.model.drop_path_prob = self.config.tab2vox_drop_path_prob * epoch / self.config.tab2vox_epoch_size

        start_time = time.time()
        for idx, (inputs, targets, keys) in enumerate(train_loader):
            
            print('augment train_batch_idx', idx)

            inputs, targets, keys = inputs.to(self.device), targets.to(self.device), keys.to(self.device)                
            logits, aux_logits = self.model(inputs) 
          
            loss = self.criterion(torch.squeeze(logits), targets)

            if self.config.tab2vox_auxiliary:
                aux_loss = self.criterion(torch.squeeze(aux_logits), targets)
                loss += self.config.tab2vox_auxiliary_weight * aux_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.tab2vox_grad_clip)
            self.optimizer.step()

            acc = accuracy(logits, targets)
            loss = loss.item()
            progress.update([loss, acc], n=inputs.size(0))
            if idx % 20 == 0:
                progress.display(idx+1)

        self.scheduler: self.scheduler.step()
        finish_time = time.time()
        epoch_time = finish_time - start_time
        progress.display(idx, f' | {epoch_time:.0f}s' + '\n')

    
    def validate(self, val_loader, epoch):
        progress = ProgressMeter(["val_loss", "val_acc"], len(val_loader), prefix=f'VALID {epoch:03d}')
        self.model.eval()
                
        with torch.no_grad():
            for idx, (inputs, targets, keys) in enumerate(val_loader):

                print('augment valid_batch_idx', idx)

                inputs, targets, keys = inputs.to(self.device), targets.to(self.device), keys.to(self.device)
                
                logits, _ = self.model(inputs)

                loss = self.criterion(torch.squeeze(logits), targets)

                acc = accuracy(logits, targets)
                progress.update([loss, acc], n=inputs.size(0))

            if progress.val_acc > self.best_acc:
                self.best_epoch = epoch
                self.best_acc = progress.val_acc
                ckpt = {
                    'best_epoch': self.best_epoch,
                    'best_acc': self.best_acc,
                    'model_state_dict': self.model.state_dict()
                }

                torch.save(ckpt, 'results/tab2vox/ckpt_test5_ep10_10_bs2_220706_16h20m.pt')

            aug_finish_time = time.time()
            aug_finish_time = strftime('%Y-%m-%d %I:%M:%S %p', localtime(aug_finish_time))
            print('aug_finish_time', aug_finish_time)

            progress.display(idx, '\n')


    def test(self, test_loader):
        progress = ProgressMeter(["test_loss", "test_acc"], len(test_loader), prefix=f'TEST')
        ckpt = torch.load('results/tab2vox/ckpt_test4_ep10_10_bs2_220706_16h35m_basis.pt')  
        self.model.load_state_dict(ckpt['model_state_dict'])

        self.model.eval()

        with torch.no_grad():

            df_test_acc = pd.DataFrame(columns=['idx', 'keys', 'logits', 'targets', 'acc'])
            for idx, (inputs, targets, keys) in enumerate(test_loader):

                print('augment test_batch_idx', idx) 

                inputs, targets, keys = inputs.to(self.device), targets.to(self.device), keys.to(self.device)
                logits, _ = self.model(inputs)  

                keys = keys[0,0,0,0,0]          
                logits = torch.squeeze(logits)
                targets = torch.squeeze(targets)

                loss = self.criterion(logits, targets)   

                acc = accuracy(logits, targets)         )
                progress.update([loss, acc], n=inputs.size(0))

                idx = torch.tensor(idx).cpu()
                keys = keys.cpu()
                logits = logits.cpu()
                targets = targets.cpu()
                acc = torch.tensor(acc).cpu()

                idx = idx.numpy()
                keys = keys.numpy()
                logits = logits.numpy()
                targets = targets.numpy()
                acc = acc.numpy()

                df_test_acc = df_test_acc.append(pd.DataFrame([[idx, keys, logits, targets, acc]], 
                                                                columns=['idx', 'keys', 'logits', 'targets', 'acc']))

            aug_finish_time = time.time()
            aug_finish_time = strftime('%Y-%m-%d %I:%M:%S %p', localtime(aug_finish_time))
            print('aug_finish_time', aug_finish_time) 
                           
            df_test_acc.to_csv('results/tab2vox/df_rslt_valid4_ep10_10_bs2_220722_11h14m.csv', header=True)
            progress.display(idx, '\n')    

