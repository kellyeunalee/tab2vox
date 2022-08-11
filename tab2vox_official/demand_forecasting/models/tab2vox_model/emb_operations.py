from collections import namedtuple
import pandas as pd
import numpy as np
from itertools import product
import itertools

import torch


clus_rslt_B = torch.load('./results/feature_clustering/clustered_bases_grp3')       
clus_rslt_E = torch.load('./results/feature_clustering/clustered_equips_grp2') 
clus_rslt_X = torch.load('./results/feature_clustering/clustered_x_feature_grp5')

df_clus_rslt_B = pd.DataFrame(clus_rslt_B, columns=['feature_name', 'grp']).astype('str')    
df_clus_rslt_E = pd.DataFrame(clus_rslt_E, columns=['feature_name', 'grp']).astype('str')    
df_clus_rslt_X = pd.DataFrame(clus_rslt_X, columns=['feature_name', 'grp']).astype('str')   
clus_rslt = [df_clus_rslt_B, df_clus_rslt_E, df_clus_rslt_X]

grp_B = list(df_clus_rslt_B['grp'].unique())
grp_E = list(df_clus_rslt_E['grp'].unique())
grp_X = list(df_clus_rslt_X['grp'].unique())
grp = [grp_B, grp_E, grp_X]

grp_cases = []
for i_feature_grp in grp:
    i_feature_cases = list(itertools.permutations(i_feature_grp, len(i_feature_grp)))
    grp_cases.append(i_feature_cases)

emb_Genotype = namedtuple('emb_Genotype', 'emb')
emb_PRIMITIVES = list(product(*grp_cases))  


emb_OPS = {}
for primitives in emb_PRIMITIVES:
    emb_OPS[str(primitives)] = lambda primitives: Tab2Vox(primitives) 


def get_col_order(f, grp_perm):
    
    grp_order = list(primitives[f]) 

    df_lst = []
    for g in grp_order:  
        d_clus_rslt = clus_rslt[f]       
        df_g = d_clus_rslt[d_clus_rslt['grp'] == g]
        df_lst.append(df_g)
    
    cat_df = pd.concat(df_lst)
    
    cat_df['col_order'] = cat_df.index
    col_order = torch.tensor(cat_df['col_order'].unique())

    return col_order


class Tab2Vox:
    def __init__(self, primitives):
        self.primitives = primitives
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def img_gen(self, init_5d):     

        features_col_order = []
        for f in range(len(self.primitives)): 
            
            f_feature_col_order = get_col_order(f, self.primitives).to(self.device) 

            torch.manual_seed(3)  
            indices = torch.randperm(f_feature_col_order.size(0)).to(self.device)   
            f_feature_col_order = torch.index_select(f_feature_col_order, dim=0, index=indices)            

            features_col_order.append(f_feature_col_order)    

        reorder_f0 = torch.index_select(init_5d, dim=2, index=features_col_order[0])        
        reorder_f1 = torch.index_select(reorder_f0, dim=3, index=features_col_order[1])     
        reorder_f2 = torch.index_select(reorder_f1, dim=4, index=features_col_order[2])     
        reorder_init_4d = reorder_f2

        return reorder_init_4d
 