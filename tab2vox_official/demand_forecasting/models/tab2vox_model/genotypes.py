
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'avg_pool_3x3x3',  
    'max_pool_3x3x3',
    'skip_connect',  
    'conv_1x1x1',
    'conv_3x3x3',
    'sep_conv_3x3x3', 
    'dil_conv_3x3x3',    
    'conv_5x5x5',
    'conv_1x3x3',
    'conv_3x1x1',
    ]
