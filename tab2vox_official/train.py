import argparse
import pandas as pd
import time
from time import localtime
from time import strftime

import torch
import torch.nn as nn
import torch.optim as optim

from demand_forecasting.data_loader_usage import get_loaders
from demand_forecasting.tab2vox_trainer import Tab2voxTrainer
from demand_forecasting.models.tab2vox_model.augment_network import Network
from demand_forecasting.models.tab2vox_model.train_search import search_main

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--train_ratio', type=float, default=.75)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--model', type=str)
    p.add_argument('--dataset', type=str, default='MND')   
    p.add_argument('--tab2vox_train_ratio', type=float, default=.5) 
    p.add_argument('--tab2vox_auxiliary', default=True)  
    p.add_argument('--tab2vox_auxiliary_weight', default=.4)      
    p.add_argument('--tab2vox_drop_path_prob', default=.2)  
    p.add_argument('--tab2vox_init_C', default=16)  
    p.add_argument('--tab2vox_n_layers', default=8)  
    p.add_argument('--tab2vox_lr', default=.025)  
    p.add_argument('--tab2vox_momentum', default=.9)  
    p.add_argument('--tab2vox_weight_decay', default=3e-4)  
    p.add_argument('--tab2vox_grad_clip', default=5)  
    p.add_argument('--tab2vox_batch_size', type=int, default=64)              
    p.add_argument('--tab2vox_epoch_size', type=int, default=10)  
    
    config = p.parse_args()

    return config


def main(config):
    
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)
    print('gpu_id', config.gpu_id)
    
    if config.model == 'tab2vox':
        train_loader, valid_loader, test_loader = get_loaders(config)
        search_main(config, train_loader, valid_loader, test_loader) 
        
        aug_start_time = time.time()
        aug_start_time = strftime('%Y-%m-%d %I:%M:%S %p', localtime(aug_start_time))
        print('aug_start_time', aug_start_time)

        tab2vox_loader = torch.load('./data/tab2vox_loader_test4_bs2')
        train_loader = tab2vox_loader['train_loader']   
        valid_loader = tab2vox_loader['valid_loader']   
        test_loader = tab2vox_loader['test_loader']  
    
    if config.model == 'tab2vox': 
        ckpt = torch.load('results/tab2vox/search_ckpt_test4_ep10_10_bs2_220706_16h35m_basis.pt')
        genotype = ckpt['genotype']
        emb_genotype = ckpt['emb_genotype']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = nn.MSELoss().to(device)
        model = Network(config.tab2vox_init_C, 10, config.tab2vox_n_layers, config.tab2vox_auxiliary, genotype, emb_genotype).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.tab2vox_epoch_size, eta_min=1e-3)

    if config.verbose >= 2:
        print(model)
        print(optimizer)
        print(crit)

    if config.model == 'tab2vox':
        
        trainer = Tab2voxTrainer(model, criterion, optimizer, scheduler, device, config)

        for ep in range(config.tab2vox_epoch_size):
            print('----- augment ep: {} -----'.format(ep))

            trainer.train(train_loader, ep)
            print('finished train phase-----------------')
            
            trainer.validate(valid_loader, ep) 
            print('finished valid phase-----------------')
        
        trainer.test(valid_loader)     
        print('finished test phase-----------------')
          
if __name__ == '__main__':
    config = define_argparser()
    main(config)
