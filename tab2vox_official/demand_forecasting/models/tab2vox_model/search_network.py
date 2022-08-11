import numpy as np
from itertools import product, permutations

import torch
import torch.nn as nn
import torch.nn.functional as F

from demand_forecasting.models.tab2vox_model.search_cell import SearchCell
from demand_forecasting.models.tab2vox_model.emb_operations import emb_PRIMITIVES, emb_Genotype
from demand_forecasting.models.tab2vox_model.genotypes import PRIMITIVES, Genotype
from demand_forecasting.models.tab2vox_model.emb_search_cell import emb_SearchCell

class SearchNetwork(nn.Module):
   
    def __init__(self, C, n_classes, n_layers, criterion, device, steps=4, multiplier=4, stem_multiplier=3):
        super(SearchNetwork, self).__init__()
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.criterion = criterion
        self.device = device
        self.steps = steps
        self.multiplier = multiplier
        
        curr_C = C * stem_multiplier    
        self.stem = nn.Sequential(
            nn.Conv3d(1, curr_C, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm3d(curr_C)
        )   

        num_emb_ops = len(emb_PRIMITIVES)   
        self.alpha_emb = nn.Parameter(torch.randn(num_emb_ops) * 1e-3)  
        self.emb_cell = emb_SearchCell(curr_C)

        prev_prev_C, prev_C, curr_C = curr_C, curr_C, C   
            
        self.cells = nn.ModuleList()
        prev_reduction = False
        
        for i in range(n_layers): 
            
            if i in [n_layers//3, 2*n_layers//3]:   
                curr_C *= 2
                reduction = True
            else:
                reduction = False   
 
            cell = SearchCell(i, steps, multiplier, prev_prev_C, prev_C, curr_C, reduction, prev_reduction)
            prev_reduction = reduction
            self.cells.append(cell) 
            
            prev_prev_C, prev_C = prev_C, multiplier * curr_C
        
        self.year_layers = torch.nn.Conv3d(in_channels = 256, out_channels = 256, 
                kernel_size = (35,4,13), stride = 4, padding=0)
        
        self.regressor = nn.Sequential( 
            nn.AdaptiveAvgPool3d(1),     
            nn.Flatten(),               
            nn.Linear(prev_C, 1)        
        )

        k = sum(1 for i in range(steps) for j in range(2 + i))
        
        num_ops = len(PRIMITIVES) 
        
        self.alpha_normal = nn.Parameter(torch.randn(k, num_ops) * 1e-3)    
        self.alpha_reduce = nn.Parameter(torch.randn(k, num_ops) * 1e-3)    
        
    def arch_parameters(self):
        return [self.alpha_normal, self.alpha_reduce, self.alpha_emb]
    
    def loss(self, inputs, targets):   
        logits = self.forward(inputs)                           
        return self.criterion(torch.squeeze(logits), targets)   

    def forward(self, x):  
        s1_lst = [] 
        for j in range(x.size(dim=3)): 
            x_j = x[:,:,:,j,:]               
            x_j = x_j.view(-1, 1, x_j.size(-3), x_j.size(-2), x_j.size(-1)) 
    
            self.emb_weights = F.softmax(self.alpha_emb, dim=-1) 

            s0_j = s1_j = self.emb_cell(x_j, self.emb_weights)    
            
            for i, cell in enumerate(self.cells):   
                if cell.reduction:
                    weights = F.softmax(self.alpha_reduce, dim=-1)
                else:
                    weights = F.softmax(self.alpha_normal, dim=-1) 
        
                s0_j, s1_j = s1_j, cell(s0_j, s1_j, weights)   
                
            s1_lst.append(s1_j)     
        
        s1_cat = torch.cat(s1_lst, dim=3)   
        s = self.year_layers(s1_cat)   

        logits = self.regressor(s)     
        return logits   

    def genotype(self):
        def _parse(weights): 
            gene = []   
            n = 2
            start = 0

            for i in range(self.steps): 
                
                end = start + n 
                W = weights[start:end].copy() 
                edges = sorted(range(i + 2), key = lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                
                for j in edges: 
                    k_best = None
                    
                    for k in range(len(W[j])): 
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:   
                                k_best = k

                    gene.append((PRIMITIVES[k_best], j)) 
                start = end
                n += 1  
            return gene

        gene_normal = _parse(F.softmax(self.alpha_normal, dim=-1).data.cpu().numpy()) 
        gene_reduce = _parse(F.softmax(self.alpha_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self.steps - self.multiplier, self.steps + 2) 

        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
    
        return genotype
    
    
    def emb_genotype(self):
        def _emb_parse(emb_weights):        
            emb_gene = []                   
            emb_W = emb_weights.copy()
            k_best = None                   
            for k in range(len(emb_W)):     
                if k_best is None or emb_W[k] > emb_W[k_best]:
                    k_best = k
              
            emb_gene.append((emb_PRIMITIVES[k_best]))
            return emb_gene
        
        emb_gene = _emb_parse(F.softmax(self.alpha_emb, dim=-1).data.cpu().numpy()) 
        emb_genotype = emb_Genotype(emb=emb_gene) 
        return emb_genotype    
       