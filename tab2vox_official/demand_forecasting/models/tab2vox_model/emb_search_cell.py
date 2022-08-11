from re import X
import torch.nn as nn
import torch

from demand_forecasting.models.tab2vox_model.emb_operations import Tab2Vox, emb_OPS, emb_PRIMITIVES


class emb_MixedOP(nn.Module):
    def __init__(self, curr_C):
        super(emb_MixedOP, self).__init__()  

        self.emb_ops = []   
        for emb_primitives in emb_PRIMITIVES:
            tab2vox = Tab2Vox(emb_primitives)   
            self.emb_ops.append(tab2vox)

        self.stem = nn.Sequential(
            nn.Conv3d(1, curr_C, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm3d(curr_C)
        )   
    
    def forward(self, x, emb_weights):
        
        emb_out = 0
        for w, emb_i in zip(emb_weights, self.emb_ops):
            img_i = emb_i.img_gen(x)    
            emb_out += w * self.stem(img_i)
            return emb_out
        
class emb_SearchCell(nn.Module):  
    def __init__(self, curr_C):
        super(emb_SearchCell, self).__init__()
        self.curr_C = curr_C  
        self.emb_op = emb_MixedOP(curr_C)
        
    def forward(self, x, emb_weights):               
        mixed_out = self.emb_op(x, emb_weights)      
        return mixed_out
