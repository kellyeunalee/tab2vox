
import torch.nn as nn

from demand_forecasting.models.tab2vox_model.operations import *
from demand_forecasting.models.tab2vox_model.emb_search_cell import emb_SearchCell
from demand_forecasting.models.tab2vox_model.emb_operations import *

def drop_path(x, drop_prob):
    if drop_prob > 0:
        keep_prob = 1 - drop_prob
        mask = x.bernoulli(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x
 
    
class Cell(nn.Module):
    def __init__(self, genotype, prev_prev_C, prev_C, C, reduction, prev_reduction):
        super(Cell, self).__init__()

        if prev_reduction:
            self.prep0 = FactorizedReduce(prev_prev_C, C)
        else:
            self.prep0 = ReLUConvBN(prev_prev_C, C, kernel_size=1, stride=1, padding=0)
        self.prep1 = ReLUConvBN(prev_C, C, kernel_size=1, stride=1, padding=0)
        
        if reduction:
            op_names, indices = zip(*genotype.reduce) 
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
                    
        self.steps = len(op_names) // 2
        self.indices = indices
        self.concat = concat
        self.multiplier = len(concat)
        self.ops = nn.ModuleList()
        
        for name, idx in zip(op_names, indices):
            stride = 2 if reduction and idx < 2 else 1
            self.stride = stride   
            op = OPS[name](C, stride, True)
            self.ops.append(op)  

    def forward(self, s0, s1, drop_prob):         
        s0 = self.prep0(s0)     
        s1 = self.prep1(s1)     

        states = [s0, s1]   
        for i in range(self.steps):
            
            h1 = states[self.indices[2 * i]]        
            h2 = states[self.indices[2 * i + 1]]
            
            op1 = self.ops[2 * i]                     
            op2 = self.ops[2 * i + 1]   
            
            h1 = op1(h1)    
            h2 = op2(h2)
                       
            if self.training and drop_prob > 0:
                if not isinstance(op1, nn.Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, nn.Identity):
                    h2 = drop_path(h2, drop_prob)   

            s = h1 + h2         
            states.append(s)    
        
        s = []
        for i in self.concat:
            s.append(states[i])
        cat_out = torch.cat(s, dim=1)

        return cat_out  


class AuxiliaryHead(nn.Module):
    def __init__(self, aux_C, n_classes):
        super(AuxiliaryHead, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(2),
            nn.Conv3d(aux_C, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 768, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(768),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(768, 1)
        )
    
    def forward(self, x):
        return self.layers(x)
    

class emb_OP(nn.Module):
    def __init__(self, curr_C, emb_primitive):
        super(emb_OP, self).__init__()
        self.emb_op = Tab2Vox(emb_primitive)
        self.stem = nn.Sequential(
            nn.Conv3d(1, curr_C, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm3d(curr_C)
        )  
    def forward(self, x):
        img_x = self.emb_op.img_gen(x)    
        emb_out = self.stem(img_x)
        return emb_out
                        
class emb_Cell(nn.Module):
    def __init__(self, curr_C, emb_genotype):
        super(emb_Cell, self).__init__()
        self.emb_op_name = tuple(emb_genotype.emb[0])  
        self.emb_op = emb_OP(curr_C, self.emb_op_name)
    def forward(self, x):
        out = self.emb_op(x)
        return out
        
class Network(nn.Module):
    def __init__(self, C, n_classes, n_layers, auxiliary, genotype, emb_genotype, stem_multiplier=3):        
        super(Network, self).__init__()

        self.n_layers = n_layers
        self.auxiliary = auxiliary

        curr_C = stem_multiplier * C
        
        self.emb_cell = emb_Cell(curr_C, emb_genotype)  

        prev_prev_C, prev_C, curr_C = curr_C, curr_C, C
        self.cells = nn.ModuleList()
        prev_reduction = False
        
        self.drop_path_prob = 0.2

        for i in range(n_layers):                   

            if i in [n_layers//3, 2*n_layers //3]:  
                curr_C *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell(genotype, prev_prev_C, prev_C, curr_C, reduction, prev_reduction)
            
            prev_reduction = reduction
            self.cells.append(cell)
            
            prev_prev_C, prev_C = prev_C, cell.multiplier * curr_C

            if i == 2 * n_layers // 3:
                aux_C = prev_C
            
        if auxiliary:
            self.aux_head = AuxiliaryHead(aux_C, n_classes)
            
        self.year_layers = torch.nn.Conv3d(in_channels = 256, out_channels = 256, 
                kernel_size = (35,4,13), stride = 4, padding=0)
        
        self.aux_year_layers = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 1, stride = 1, padding=0),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9, 1)
        )
        
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(prev_C, 1)
        )
    

    def forward(self, x):   
        s1_lst = [] 
        aux_logits_lst = []
        for j in range(x.size(dim=3)): 
            
            x_j = x[:,:,:,j,:]    
            x_j = x_j.view(-1, 1, x_j.size(-3), x_j.size(-2), x_j.size(-1))   

            aux_logits = None
            
            s0 = s1 = self.emb_cell(x_j)  

            for i, cell in enumerate(self.cells):
                s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
                     
                if i == 2 * self.n_layers // 3:
                    if self.auxiliary and self.training:
                        aux_logits = self.aux_head(s1)  

            s1_lst.append(s1)
            
            if aux_logits != None: 
                aux_logits_lst.append(aux_logits)
         
        s1_cat = torch.cat(s1_lst, dim=3) 
        s = self.year_layers(s1_cat) 
        
        if len(aux_logits_lst) != 0:
            aux_logits_cat = torch.cat(aux_logits_lst, dim=1) 
            aux_logits_cat = aux_logits_cat.unsqueeze(dim=1)    
            aux = self.aux_year_layers(aux_logits_cat)          
            aux = aux.squeeze()                                 
            aux_logits = aux
            
        logits = self.regressor(s)  
        
        return logits, aux_logits
    