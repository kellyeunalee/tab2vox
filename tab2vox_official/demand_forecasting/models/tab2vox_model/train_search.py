
from IPython.display import display, Image
import time
from time import localtime
from time import strftime
import argparse
from graphviz import Digraph
from PIL import Image

import torch
import torch.nn as nn

from demand_forecasting.models.tab2vox_model.tab2vox_utils import *
from demand_forecasting.models.tab2vox_model.architecture import Architecture
from demand_forecasting.data_loader_usage import get_loaders
from demand_forecasting.models.tab2vox_model.search_network import SearchNetwork


def plot(genotype, filename):
    g = Digraph(
        format='png',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j - 2)
            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    g.render(filename, view=True)
    
    
class SearchTrainer(object):
    def __init__(self, model, architecture, criterion, optimizer, scheduler, device):
        self.model = model
        self.architecture = architecture
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device


    def train(self, train_loader, valid_loader, epoch, config):

        progress = ProgressMeter(["train_loss", "train_acc"], len(train_loader), prefix=f'EPOCH {epoch:03d}')
        self.model.train()  
        valid_iter = iter(valid_loader)
        lr = self.scheduler.get_last_lr()[0]
        
        start_time = time.time()
        
        for idx, (inputs, targets, keys) in enumerate(train_loader):  
            
            print('train_batch_idx', idx)
            inputs, targets, keys = inputs.to(self.device), targets.to(self.device), keys.to(self.device)
    
            valid_inputs, valid_targets, valid_keys = next(valid_iter)          
            valid_inputs, valid_targets, valid_keys = valid_inputs.to(self.device), valid_targets.to(self.device), valid_keys.to(self.device)
            
            self.architecture.step(valid_inputs, valid_targets)

            logits = self.model(inputs)                            

            loss = self.criterion(torch.squeeze(logits), targets)

            self.optimizer.zero_grad()                              
            loss.backward()                                         
            nn.utils.clip_grad_norm_(self.model.parameters(), config.tab2vox_grad_clip)
            self.optimizer.step()                                  

            acc = accuracy(logits, targets)
            loss = loss.item()                                      
            progress.update([loss, acc], n=inputs.size(0))
            if idx % 5 == 0:
                progress.display(idx)

        self.scheduler.step()
        finish_time = time.time()
        epoch_time = finish_time - start_time
        progress.display(idx, f' | {epoch_time:.0f}s' + '\n')


    def validate(self, val_loader, epoch):
        self.model.eval()

        with torch.no_grad():
            for idx, (inputs, targets, keys) in enumerate(val_loader): 

                print('valid_batch_idx', idx)

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits = self.model(inputs)
                
                loss = self.criterion(torch.squeeze(logits), targets)

                acc = accuracy(logits, targets)
        
        genotype = self.model.genotype()

        search_finish_time = time.time()
        search_finish_time = strftime('%Y-%m-%d %I:%M:%S %p', localtime(search_finish_time))
        print('search_finish_time', search_finish_time)

        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'genotype': self.model.genotype(),
            'emb_genotype': self.model.emb_genotype()
        }
        
        torch.save(ckpt, 'results/tab2vox/search_ckpt_test5_ep10_10_bs2_220706_16h20m.pt')
 
        
def search_main(config, train_loader, valid_loader, test_loader): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    criterion = nn.MSELoss().to(device) 
    model = SearchNetwork(config.tab2vox_init_C, 10, config.tab2vox_n_layers, criterion, device).to(device)
    optimizer = torch.optim.AdamW(model.arch_parameters(), lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.tab2vox_epoch_size, eta_min=1e-3)
    architecture = Architecture(model)
    trainer = SearchTrainer(model, architecture, criterion, optimizer, scheduler, device)

    for ep in range(config.tab2vox_epoch_size):
        print('----- search ep: {} -----'.format(ep))
        
        trainer.train(train_loader, valid_loader, ep, config)  
        print('finished search train phase -------------------') 
        
        trainer.validate(valid_loader, ep)
        print('finished search valid phase -------------------')     

    # print('Normal Cell')
    # display(Image.open(f'results/tab2vox/{ep:02d}_normal.png')) 
    # print('Reduce Cell')
    # display(Image.open(f'results/tab2vox/{ep:02d}_reduce.png'))