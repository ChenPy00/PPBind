# -*- coding: utf-8 -*-
import random
import torch

from ._base import _index_select_data, register_transform, _get_CB_positions


@register_transform('focused_random_patch')
class FocusedRandomPatch(object):

    def __init__(self, focus_attr, seed_nbh_size=32, patch_size=128):
        super().__init__()
        self.focus_attr = focus_attr
        self.seed_nbh_size = seed_nbh_size
        self.patch_size = patch_size

    def __call__(self, data):
        focus_flag = (data[self.focus_attr] > 0)    # (L, )
        if focus_flag.sum() == 0:
            # If there is no active residues, randomly pick one.
            focus_flag[random.randint(0, focus_flag.size(0)-1)] = True
        seed_idx = torch.multinomial(focus_flag.float(), num_samples=1).item()

        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])   # (L, )
        pos_seed = pos_CB[seed_idx:seed_idx+1]  # (1, )
        dist_from_seed = torch.cdist(pos_CB, pos_seed)[:, 0]    # (L, 1) -> (L, )
        nbh_seed_idx = dist_from_seed.argsort()[:self.seed_nbh_size]    # (Nb, )

        core_idx = nbh_seed_idx[focus_flag[nbh_seed_idx]]  # (Ac, ), the core-set must be a subset of the focus-set
        dist_from_core = torch.cdist(pos_CB, pos_CB[core_idx]).min(dim=1)[0]    # (L, )
        patch_idx = dist_from_core.argsort()[:self.patch_size]    # (P, )
        patch_idx = patch_idx.sort()[0]

        core_flag = torch.zeros([data['aa'].size(0), ], dtype=torch.bool)
        core_flag[core_idx] = True
        data['core_flag'] = core_flag

        data_patch = _index_select_data(data, patch_idx)
        return data_patch


@register_transform('random_patch')
class RandomPatch(object):

    def __init__(self, seed_nbh_size=32, patch_size=128):
        super().__init__()
        self.seed_nbh_size = seed_nbh_size
        self.patch_size = patch_size

    def __call__(self, data):
        seed_idx = random.randint(0, data['aa'].size(0)-1)

        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])   # (L, )
        pos_seed = pos_CB[seed_idx:seed_idx+1]  # (1, )
        dist_from_seed = torch.cdist(pos_CB, pos_seed)[:, 0]    # (L, 1) -> (L, )
        core_idx = dist_from_seed.argsort()[:self.seed_nbh_size]    # (Nb, )

        dist_from_core = torch.cdist(pos_CB, pos_CB[core_idx]).min(dim=1)[0]    # (L, )
        patch_idx = dist_from_core.argsort()[:self.patch_size]    # (P, )
        patch_idx = patch_idx.sort()[0]

        core_flag = torch.zeros([data['aa'].size(0), ], dtype=torch.bool)
        core_flag[core_idx] = True
        data['core_flag'] = core_flag

        data_patch = _index_select_data(data, patch_idx)
        return data_patch



@register_transform('selected_region_with_padding_patch')
class SelectedRegionWithPaddingPatch(object):

    def __init__(self, select_attr, each_residue_nbh_size, patch_size_limit):
        super().__init__()
        self.select_attr = select_attr
        self.each_residue_nbh_size = each_residue_nbh_size
        self.patch_size_limit = patch_size_limit
    
    def __call__(self, data):
        select_flag = (data[self.select_attr] > 0)

        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])   # (L, 3)
        pos_sel = pos_CB[select_flag]   # (S, 3)
        dist_from_sel = torch.cdist(pos_CB, pos_sel)    # (L, S)
        nbh_sel_idx = torch.argsort(dist_from_sel, dim=0)[:self.each_residue_nbh_size, :]  # (nbh, S)
        patch_idx = nbh_sel_idx.view(-1).unique()       # (patchsize,)

        data_patch = _index_select_data(data, patch_idx)
        return data_patch


# +
# @register_transform('selected_region_fixed_size_patch')
# class SelectedRegionFixedSizePatch(object):

#     def __init__(self, select_attr, patch_size):
#         super().__init__()
#         # 若选择突变位点为中心，则select_attr取值为'mut_flag'
#         self.select_attr = select_attr
#         self.patch_size = patch_size
    
#     def __call__(self, data):
#         select_flag = (data[self.select_attr] > 0)

#         # pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])   # (L, 3)
#         # pos_sel = pos_CB[select_flag]   # (S, 3)
#         # dist_from_sel = torch.cdist(pos_CB, pos_sel).min(dim=1)[0]    # (L, )

#         pos_CA = data['pos_atoms'][:, 1, :]
#         pos_sel = pos_CA[select_flag]
#         dist_from_sel = torch.cdist(pos_CA, pos_sel).min(dim=1)[0]

#         # patch_idx = torch.argsort(dist_from_sel)[:self.patch_size]
#         # data_patch = _index_select_data(data, patch_idx)

#         # ligand跟receptor各取一半
#         ligand_idx = torch.where(data['group_id'] == 1)[0]
#         ligand_patch_idx = torch.argsort(dist_from_sel[ligand_idx])[:int(self.patch_size/2)]
#         ligand_patch_idx = ligand_idx[ligand_patch_idx]
#         receptor_idx = torch.where(data['group_id'] == 2)[0]
#         receptor_patch_idx = torch.argsort(dist_from_sel[receptor_idx])[:int(self.patch_size/2)]
#         receptor_patch_idx = receptor_idx[receptor_patch_idx]
#         patch_idx = torch.cat((ligand_patch_idx, receptor_patch_idx), dim=0)
#         data_patch = _index_select_data(data, patch_idx)

#         return data_patch
# -

# 20240426
@register_transform('selected_region_fixed_size_patch')
class SelectedRegionFixedSizePatch(object):

    def __init__(self, select_attr, patch_size):
        super().__init__()
        self.select_attr = select_attr
        self.patch_size = patch_size
    
    def __call__(self, data):
        select_flag = (data[self.select_attr] > 0)

        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])   # (L, 3)
        pos_sel = pos_CB[select_flag]   # (S, 3)
        dist_from_sel = torch.cdist(pos_CB, pos_sel).min(dim=1)[0]    # (L, )
        patch_idx = torch.argsort(dist_from_sel)[:self.patch_size]

        data_patch = _index_select_data(data, patch_idx)

        return data_patch


# 20240731: 选择结合界面附近的片段（不连续）
@register_transform('selected_region_seperated_fixed_size_patch')
class SelectedRegionSeperatedFixedSizePatch(object):

    def __init__(self, select_attr, patch_size):
        super().__init__()
        self.select_attr = select_attr
        self.patch_size = patch_size
    
    def __call__(self, data):
        select_flag = (data[self.select_attr] > 0)

        # 计算距离
        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])# (L, 3)
        pos_sel = pos_CB[select_flag]   # (S, 3)
        dist_from_sel = torch.cdist(pos_CB, pos_sel).min(dim=1)[0]# (L, )
        
        # receptor:
        ## 选出receptor
        receptor_idx = torch.where(data['group_id'] == 2)[0]
        data_receptor = _index_select_data(data, receptor_idx)
        ## 选出receptor上的KNN残基
        dist_from_sel_receptor = dist_from_sel[receptor_idx]# 受体上每个残基与结合界面的最近距离
        patch_idx_receptor = torch.argsort(dist_from_sel_receptor)[:self.patch_size]# 截取受体上与结合界面最近的k=self.patch_size个残基
        patch_idx_receptor,_ = patch_idx_receptor.sort(descending=False)# 恢复残基在序列的原有顺序
        data_receptor = _index_select_data(data_receptor, patch_idx_receptor)
        
        
        # ligand:
        ## 选出ligand
        ligand_idx = torch.where(data['group_id'] == 1)[0]
        data_ligand = _index_select_data(data, ligand_idx)
        ## 选出ligand上KNN残基
        dist_from_sel_ligand = dist_from_sel[ligand_idx]# 受体上每个残基与结合界面的最近距离
        patch_idx_ligand = torch.argsort(dist_from_sel_ligand)[:self.patch_size]# 截取配体上与结合界面最近的k=self.patch_size个残基
        patch_idx_ligand,_ = patch_idx_ligand.sort(descending=False)# 恢复残基在序列的原有顺序
        data_ligand = _index_select_data(data_ligand, patch_idx_ligand)
        
        return {'receptor':data_receptor, 'ligand':data_ligand}


# +
# 20240716: 选择结合界面附近的连续片段
@register_transform('separate_receptor_ligand')
class SeparateReceptorLigand(object):

    def __init__(self,):
        super().__init__()
    
    def __call__(self, data):        
        # receptor:
        ## 选出receptor
        receptor_idx = torch.where(data['group_id'] == 2)[0]
        data_receptor = _index_select_data(data, receptor_idx)        
        
        # ligand:
        ## 选出ligand
        ligand_idx = torch.where(data['group_id'] == 1)[0]
        data_ligand = _index_select_data(data, ligand_idx)
        
        return {'receptor':data_receptor, 'ligand':data_ligand}

    
def get_knn_contiguous_sequences(data, patch_idx, patch_size):
    ## 逐条链链截取连续的片段（以该链KNN位点为中心，截取self.path_size的连续片段,若KNN范围大于path_size，则往大范围截取）
    continue_patch_idx = []
    for chain_nb in data['chain_nb'].unique():
        # 属于chain_nb链且是结合界面KNN的残基的序号
        patch_idx_chain_nb = set(torch.where(data['chain_nb']==chain_nb)[0].tolist()).intersection(set(patch_idx.tolist()))
        if len(patch_idx_chain_nb) > 0:
            start_KNN = min(patch_idx_chain_nb)
            end_KNN = max(patch_idx_chain_nb)
            min_idx_chain_nb = torch.where(data['chain_nb']==chain_nb)[0].min()
            max_idx_chain_nb = torch.where(data['chain_nb']==chain_nb)[0].max()
            idx_middle = int((start_KNN+end_KNN)/2)
            start = max([min_idx_chain_nb,min([start_KNN, idx_middle-patch_size//2])])
            end = min([max_idx_chain_nb,max([end_KNN, idx_middle+patch_size//2])])
            continue_patch_idx += list(range(start,end+1))
    continue_patch_idx,_ = torch.tensor(continue_patch_idx).sort(descending=False)# 恢复残基在序列的原有顺序
    data = _index_select_data(data, continue_patch_idx)
    return data


# 20240716: 选择结合界面附近的连续片段
@register_transform('selected_region_continued_fixed_size_patch')
class SelectedRegionContinuedFixedSizePatch(object):

    def __init__(self, select_attr, patch_size):
        super().__init__()
        self.select_attr = select_attr
        self.patch_size = patch_size
    
    def __call__(self, data):
        select_flag = (data[self.select_attr] > 0)

        # 计算距离
        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])# (L, 3)
        pos_sel = pos_CB[select_flag]   # (S, 3)
        dist_from_sel = torch.cdist(pos_CB, pos_sel).min(dim=1)[0]# (L, )
        
        # receptor:
        ## 选出receptor
        receptor_idx = torch.where(data['group_id'] == 2)[0]
        data_receptor = _index_select_data(data, receptor_idx)
        ## 选出receptor上的KNN残基
        dist_from_sel_receptor = dist_from_sel[receptor_idx]# 受体上每个残基与结合界面的最近距离
        patch_idx_receptor = torch.argsort(dist_from_sel_receptor)[:self.patch_size]# 截取受体上与结合界面最近的k=self.patch_size个残基
        patch_idx_receptor,_ = patch_idx_receptor.sort(descending=False)# 恢复残基在序列的原有顺序
        ## 逐条链截取连续的片段（以该链KNN位点为中心，截取self.path_size的连续片段）
        data_receptor = get_knn_contiguous_sequences(data_receptor, patch_idx_receptor, self.patch_size)
        
        
        # ligand:
        ## 选出ligand
        ligand_idx = torch.where(data['group_id'] == 1)[0]
        data_ligand = _index_select_data(data, ligand_idx)
        ## 选出ligand上KNN残基的连续片段
        dist_from_sel_ligand = dist_from_sel[ligand_idx]# 受体上每个残基与结合界面的最近距离
        patch_idx_ligand = torch.argsort(dist_from_sel_ligand)[:self.patch_size]# 截取配体上与结合界面最近的k=self.patch_size个残基
        patch_idx_ligand,_ = patch_idx_ligand.sort(descending=False)# 恢复残基在序列的原有顺序
        ## 逐条链截取连续的片段（以该链KNN位点为中心，截取self.path_size的连续片段）
        data_ligand = get_knn_contiguous_sequences(data_ligand, patch_idx_ligand, self.patch_size)
        
        return {'receptor':data_receptor, 'ligand':data_ligand}
# +



# 202400807: 选择结合界面附近的包含所有knn位点的最短连续片段
@register_transform('selected_region_continued_patch')
class SelectedRegionContinuedPatch(object):

    def __init__(self, select_attr, patch_size):
        super().__init__()
        self.select_attr = select_attr
        self.patch_size = patch_size
        
    def get_contiguous_sequences(self, data, patch_idx):
        ## 逐条链截取连续的片段（截取包含patch_idx的最短连续片段）
        continue_patch_idx = []
        for chain_nb in data['chain_nb'].unique():
            # 属于chain_nb链且是patch_idx残基的序号
            patch_idx_chain_nb = set(torch.where(data['chain_nb']==chain_nb)[0].tolist()).intersection(set(patch_idx.tolist()))
            if len(patch_idx_chain_nb) > 0:
                start = min(patch_idx_chain_nb)
                end = max(patch_idx_chain_nb)
                continue_patch_idx += list(range(start,end+1))
        continue_patch_idx,_ = torch.tensor(continue_patch_idx).sort(descending=False)# 恢复残基在序列的原有顺序
        data = _index_select_data(data, continue_patch_idx)
        return data
    
    def __call__(self, data):
        select_flag = (data[self.select_attr] > 0)

        # 计算距离
        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])# (L, 3)
        pos_sel = pos_CB[select_flag]   # (S, 3)
        dist_from_sel = torch.cdist(pos_CB, pos_sel).min(dim=1)[0]# (L, )
        
        # receptor:
        ## 选出receptor
        receptor_idx = torch.where(data['group_id'] == 2)[0]
        data_receptor = _index_select_data(data, receptor_idx)
        ## 选出receptor上的KNN残基
        dist_from_sel_receptor = dist_from_sel[receptor_idx]# 受体上每个残基与结合界面的最近距离
        patch_idx_receptor = torch.argsort(dist_from_sel_receptor)[:self.patch_size]# 截取受体上与结合界面最近的k=self.patch_size个残基
        patch_idx_receptor,_ = patch_idx_receptor.sort(descending=False)# 恢复残基在序列的原有顺序
        ## 逐条链截取连续的片段（以该链KNN位点为中心，截取self.path_size的连续片段）
        data_receptor = self.get_contiguous_sequences(data_receptor, patch_idx_receptor)
        
        # ligand:
        ## 选出ligand
        ligand_idx = torch.where(data['group_id'] == 1)[0]
        data_ligand = _index_select_data(data, ligand_idx)
        ## 选出ligand上KNN残基的连续片段
        dist_from_sel_ligand = dist_from_sel[ligand_idx]# 受体上每个残基与结合界面的最近距离
        patch_idx_ligand = torch.argsort(dist_from_sel_ligand)[:self.patch_size]# 截取配体上与结合界面最近的k=self.patch_size个残基
        patch_idx_ligand,_ = patch_idx_ligand.sort(descending=False)# 恢复残基在序列的原有顺序
        ## 逐条链截取连续的片段（包含该链KNN位点的连续片段）
        data_ligand = self.get_contiguous_sequences(data_ligand, patch_idx_ligand)
        return {'receptor':data_receptor, 'ligand':data_ligand}


# +


# 202400829: 对于receptor， 截取结合界面附近的knn位点；对于ligand, 截取包含起结合界面所有knn位点的最短连续片段
@register_transform('selected_region_mix_patch')
class SelectedRegionMixPatch(object):

    def __init__(self, select_attr, patch_size):
        super().__init__()
        self.select_attr = select_attr
        self.patch_size = patch_size
        
    def get_contiguous_sequences(self, data, patch_idx):
        ## 逐条链截取连续的片段（截取包含patch_idx的最短连续片段）
        continue_patch_idx = []
        for chain_nb in data['chain_nb'].unique():
            # 属于chain_nb链且是patch_idx残基的序号
            patch_idx_chain_nb = set(torch.where(data['chain_nb']==chain_nb)[0].tolist()).intersection(set(patch_idx.tolist()))
            if len(patch_idx_chain_nb) > 0:
                start = min(patch_idx_chain_nb)
                end = max(patch_idx_chain_nb)
                continue_patch_idx += list(range(start,end+1))
        continue_patch_idx,_ = torch.tensor(continue_patch_idx).sort(descending=False)# 恢复残基在序列的原有顺序
        data = _index_select_data(data, continue_patch_idx)
        return data
    
    def __call__(self, data):
        select_flag = (data[self.select_attr] > 0)

        # 计算距离
        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])# (L, 3)
        pos_sel = pos_CB[select_flag]   # (S, 3)
        dist_from_sel = torch.cdist(pos_CB, pos_sel).min(dim=1)[0]# (L, )        
        
        # receptor:
        ## 选出receptor
        receptor_idx = torch.where(data['group_id'] == 2)[0]
        data_receptor = _index_select_data(data, receptor_idx)
        ## 选出receptor上的KNN残基
        dist_from_sel_receptor = dist_from_sel[receptor_idx]# 受体上每个残基与结合界面的最近距离
        patch_idx_receptor = torch.argsort(dist_from_sel_receptor)[:self.patch_size]# 截取受体上与结合界面最近的k=self.patch_size个残基
        patch_idx_receptor,_ = patch_idx_receptor.sort(descending=False)# 恢复残基在序列的原有顺序
        data_receptor = _index_select_data(data_receptor, patch_idx_receptor)
        
        
        # ligand:
        ## 选出ligand
        ligand_idx = torch.where(data['group_id'] == 1)[0]
        data_ligand = _index_select_data(data, ligand_idx)
        ## 选出ligand上KNN残基的连续片段
        dist_from_sel_ligand = dist_from_sel[ligand_idx]# 受体上每个残基与结合界面的最近距离
        patch_idx_ligand = torch.argsort(dist_from_sel_ligand)[:self.patch_size]# 截取配体上与结合界面最近的k=self.patch_size个残基
        patch_idx_ligand,_ = patch_idx_ligand.sort(descending=False)# 恢复残基在序列的原有顺序
        ## 逐条链截取连续的片段（包含该链KNN位点的连续片段）
        data_ligand = self.get_contiguous_sequences(data_ligand, patch_idx_ligand)
        return {'receptor':data_receptor, 'ligand':data_ligand}

# -

# 20240731: 选择结合界面附近的片段（不连续）
@register_transform('selected_region_seperated_fixed_size_excludeAA+ESM_patch')
class SelectedRegionSeperatedFixedSizePatch(object):

    def __init__(self, select_attr, patch_size):
        super().__init__()
        self.select_attr = select_attr
        self.patch_size = patch_size
    
    def __call__(self, data):
        select_flag = (data[self.select_attr] > 0)

        # 计算距离
        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])# (L, 3)
        pos_sel = pos_CB[select_flag]   # (S, 3)
        dist_from_sel = torch.cdist(pos_CB, pos_sel).min(dim=1)[0]# (L, )
        
        # receptor:
        ## 选出receptor
        receptor_idx = torch.where(data['group_id'] == 2)[0]
        data_receptor = _index_select_data(data, receptor_idx)
        receptor_esm_embedding = data_receptor['esm_embedding']
        receptor_aa = data_receptor['aa']
        ## 选出receptor上的KNN残基
        dist_from_sel_receptor = dist_from_sel[receptor_idx]# 受体上每个残基与结合界面的最近距离
        patch_idx_receptor = torch.argsort(dist_from_sel_receptor)[:self.patch_size]# 截取受体上与结合界面最近的k=self.patch_size个残基
        patch_idx_receptor,_ = patch_idx_receptor.sort(descending=False)# 恢复残基在序列的原有顺序
        data_receptor = _index_select_data(data_receptor, patch_idx_receptor)
        data_receptor['origin_aa'] = receptor_aa
        data_receptor['esm_embedding'] = receptor_esm_embedding
        data_receptor['type'] = 'receptor'
        
        # ligand:
        ## 选出ligand
        ligand_idx = torch.where(data['group_id'] == 1)[0]
        data_ligand = _index_select_data(data, ligand_idx)
        ligand_esm_embedding = data_ligand['esm_embedding']
        ligand_aa = data_ligand['aa']
        ## 选出ligand上KNN残基
        dist_from_sel_ligand = dist_from_sel[ligand_idx]# 受体上每个残基与结合界面的最近距离
        patch_idx_ligand = torch.argsort(dist_from_sel_ligand)[:self.patch_size]# 截取配体上与结合界面最近的k=self.patch_size个残基
        patch_idx_ligand,_ = patch_idx_ligand.sort(descending=False)# 恢复残基在序列的原有顺序
        data_ligand = _index_select_data(data_ligand, patch_idx_ligand)
        data_ligand['origin_aa'] = ligand_aa
        data_ligand['esm_embedding'] = ligand_esm_embedding
        data_ligand['type'] = 'ligand'
        
        return {'receptor':data_receptor, 'ligand':data_ligand}
