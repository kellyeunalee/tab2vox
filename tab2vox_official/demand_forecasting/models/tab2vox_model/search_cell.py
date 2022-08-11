from xml.etree.ElementPath import ops
from demand_forecasting.models.tab2vox_model.genotypes import PRIMITIVES
from demand_forecasting.models.tab2vox_model.operations import *
        
class MixedOp(nn.Module):
    def __init__(self, C, stride):  
        super(MixedOp, self).__init__()

        self.ops = nn.ModuleList()

        for primitive in PRIMITIVES:    
            op = OPS[primitive](C, stride, False) 
            
            if 'pool' in primitive:
                op = nn.Sequential(
                    op,
                    nn.BatchNorm3d(C, affine=False) 
                ) 
                
            self.ops.append(op)
    
    def forward(self, x, weights): 
        out = 0
        for w, l in zip(weights, self.ops):
            out += w * l(x)
            return out

class SearchCell(nn.Module):
    def __init__(self, i, steps, multiplier, prev_prev_C, prev_C, curr_C, reduction, prev_reduction):
        super(SearchCell, self).__init__()
        self.i = i
        self.steps = steps
        self.multiplier = multiplier
        self.reduction = reduction
        self.prev_prev_C = prev_prev_C
        self.prev_C = prev_C
        self.curr_C = curr_C
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if prev_reduction:
            self.prep0 = FactorizedReduce(prev_prev_C, curr_C, affine=False) 
        else:
            self.prep0 = ReLUConvBN(prev_prev_C, curr_C, kernel_size=1, stride=1, padding=0, affine=False)
        self.prep1 = ReLUConvBN(prev_C, curr_C, kernel_size=1, stride=1, padding=0, affine=False)
      
        self.layers = nn.ModuleList()
        for i in range(steps):          
            for j in range(2+i):        
                stride = 2 if reduction and j < 2 else 1    
                op = MixedOp(curr_C, stride) 
                self.layers.append(op)
  
    def forward(self, s0, s1, weights):
        s0 = self.prep0(s0)
        s1 = self.prep1(s1) 
        states = [s0, s1]
        offset = 0

        for i in range(self.steps):                  
            s = []
            for j, h in enumerate(states):   
                mixed_out = self.layers[offset + j](h, weights[offset + j]) 
                s.append(mixed_out)
                
            sum_s = sum(s)

            offset += len(states)
            states.append(sum_s)    

        cat_out = torch.cat(states[-self.multiplier:], dim=1) 

        return cat_out