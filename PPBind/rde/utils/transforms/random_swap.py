import random
from ._base import register_transform

# 20240716: 随机调转ligand和receptor
@register_transform('randomly_swap_receptor_ligand')
class RandomlySwapReceptorLigand(object):

    def __init__(self, rate):
        super().__init__()
        self.rate = rate
    
    def __call__(self, data):
        if random.random() < self.rate:
            # original group id
            idx_ligand = data['group_id'] == 1# ligand
            idx_receptor = data['group_id'] == 2# receptor
            # swap
            data['group_id'][idx_ligand] = 2
            data['group_id'][idx_receptor] = 1
        
        return data
