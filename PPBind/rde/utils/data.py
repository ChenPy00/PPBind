# -*- coding: utf-8 -*-
import math
import torch
from torch.utils.data._utils.collate import default_collate
from torch.nn import functional as F
from typing import List, Tuple


DEFAULT_PAD_VALUES = {
    'aa': 21, 
    'aa_masked': 21,
    'aa_true': 21,
    'chain_nb': -1, 
    'pos14': 0.0,
    'chain_id': ' ', 
    'icode': ' ',
    'aa_ligand': 21,
    'aa_receptor': 21,
    'chain_nb_ligand': 0, 
    'chain_nb_receptor': 0,
    'origin_aa': 21 
}


class PaddingCollate_struc(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys


    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        keys = self._get_common_keys(data_list)
        
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:
            try:
                data_padded = {
                    k: self._pad_last(v, max_length, value=self._get_pad_value(k))
                    for k, v in data.items()
                    if k in keys
                }
            except:
                import pdb;pdb.set_trace()
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
        batch = default_collate(data_list_padded)
        batch['size'] = len(data_list_padded)
        return batch


# 只考虑二/三/四聚体结构，其他赋予mask
class PaddingCollate_mix_v2(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys


    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        
        receptor_data_list = [d['receptor'] for d in data_list]
        ligand_data_list = [d['ligand'] for d in data_list]
        
        keys = {k for k in self._get_common_keys(receptor_data_list) if 'origin' not in k and 'esm' not in k}
        rec_max_length = max([data[self.length_ref_key].size(0) for data in receptor_data_list])
        lig_max_length = max([data[self.length_ref_key].size(0) for data in ligand_data_list])
        if self.eight:
            rec_max_length = math.ceil(rec_max_length / 8) * 8
            lig_max_length = math.ceil(lig_max_length / 8) * 8
        
        
        # 结构模型要用到的特征
        struc_rec_list_padded=[]
        for data in receptor_data_list:
            data_padded = {
                k: self._pad_last(v, rec_max_length, value=self._get_pad_value(k))
                for k, v in data.items()
                if k in keys
            }
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), rec_max_length)
            struc_rec_list_padded.append(data_padded)
            
        struc_lig_list_padded=[]
        for data in ligand_data_list:
            data_padded = {
                k: self._pad_last(v, lig_max_length, value=self._get_pad_value(k))
                for k, v in data.items()
                if k in keys
            }
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), lig_max_length)
            struc_lig_list_padded.append(data_padded)
        
        
        # 序列模型要用到的特征
        rec_max_length = max([data['origin_aa'].size(0) for data in receptor_data_list])
        lig_max_length = max([data['origin_aa'].size(0) for data in ligand_data_list])
        
        seq_rec_list_padded = []
        for data in receptor_data_list:
            if data['origin_aa'].size(0)<rec_max_length:                
                pad_length = rec_max_length-data['origin_aa'].size(0)
                esm_embedding =  F.pad(data['esm_embedding'], 
                                                  pad=(0, 0, 0, pad_length), 
                                                  mode='constant', value=0)
                origin_aa = torch.cat([
                    data['origin_aa'], 
                    torch.tensor([DEFAULT_PAD_VALUES['origin_aa']] * pad_length),
                ], dim=0)
            else:
                esm_embedding = data['esm_embedding']
                origin_aa = data['origin_aa']
            seq_mask = torch.cat([
                torch.ones([data['origin_aa'].size(0)], dtype=torch.bool),
                torch.zeros([rec_max_length-data['origin_aa'].size(0)], dtype=torch.bool)
            ], dim=0)
            seq_rec_list_padded.append({'esm_embedding': esm_embedding.float(), 
                                              'origin_aa': origin_aa,
                                              'seq_mask': seq_mask, })
            
        seq_lig_list_padded = []
        for data in ligand_data_list:
            if data['origin_aa'].size(0)<lig_max_length:                
                pad_length = lig_max_length-data['origin_aa'].size(0)
                esm_embedding =  F.pad(data['esm_embedding'], 
                                                  pad=(0, 0, 0, pad_length), 
                                                  mode='constant', value=0)
                origin_aa = torch.cat([
                    data['origin_aa'], 
                    torch.tensor([DEFAULT_PAD_VALUES['origin_aa']] * pad_length),
                ], dim=0)
            else:
                esm_embedding = data['esm_embedding']
                origin_aa = data['origin_aa']
            seq_mask = torch.cat([
                torch.ones([data['origin_aa'].size(0)], dtype=torch.bool),
                torch.zeros([lig_max_length-data['origin_aa'].size(0)], dtype=torch.bool)
            ], dim=0)
            seq_lig_list_padded.append({'esm_embedding': esm_embedding.float(), 
                                              'origin_aa': origin_aa,
                                              'seq_mask': seq_mask, })
        
        
        receptor_data_batch = [{**d1, **d2} for d1, d2 in zip(struc_rec_list_padded, seq_rec_list_padded)]
        receptor_data_batch = default_collate(receptor_data_batch)
        receptor_data_batch['size'] = len(receptor_data_batch)
        
        ligand_data_batch = [{**d1, **d2} for d1, d2 in zip(struc_lig_list_padded, seq_lig_list_padded)]
        ligand_data_batch = default_collate(ligand_data_batch)
        ligand_data_batch['size'] = len(ligand_data_batch)
        
        return {
            'receptor': receptor_data_batch,
            'ligand':ligand_data_batch
        }


# 只考虑二/三/四聚体结构，其他赋予mask
class PaddingCollate_seq(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys


    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        receptor_data_list = [d['receptor'] for d in data_list]
        ligand_data_list = [d['ligand'] for d in data_list]        
        
        # 序列模型要用到的特征
        rec_max_length = max([data['origin_aa'].size(0) for data in receptor_data_list])
        lig_max_length = max([data['origin_aa'].size(0) for data in ligand_data_list])
        
        seq_rec_list_padded = []
        for data in receptor_data_list:
            if data['origin_aa'].size(0)<rec_max_length:                
                pad_length = rec_max_length-data['origin_aa'].size(0)
                esm_embedding =  F.pad(data['esm_embedding'], 
                                                  pad=(0, 0, 0, pad_length), 
                                                  mode='constant', value=0)
                origin_aa = torch.cat([
                    data['origin_aa'], 
                    torch.tensor([DEFAULT_PAD_VALUES['origin_aa']] * pad_length),
                ], dim=0)
            else:
                esm_embedding = data['esm_embedding']
                origin_aa = data['origin_aa']
            seq_mask = torch.cat([
                torch.ones([data['origin_aa'].size(0)], dtype=torch.bool),
                torch.zeros([rec_max_length-data['origin_aa'].size(0)], dtype=torch.bool)
            ], dim=0)
            seq_rec_list_padded.append({'esm_embedding': esm_embedding.float(), 
                                              'origin_aa': origin_aa,
                                              'seq_mask': seq_mask, })
            
        seq_lig_list_padded = []
        for data in ligand_data_list:
            if data['origin_aa'].size(0)<lig_max_length:                
                pad_length = lig_max_length-data['origin_aa'].size(0)
                esm_embedding =  F.pad(data['esm_embedding'], 
                                                  pad=(0, 0, 0, pad_length), 
                                                  mode='constant', value=0)
                origin_aa = torch.cat([
                    data['origin_aa'], 
                    torch.tensor([DEFAULT_PAD_VALUES['origin_aa']] * pad_length),
                ], dim=0)
            else:
                esm_embedding = data['esm_embedding']
                origin_aa = data['origin_aa']
            seq_mask = torch.cat([
                torch.ones([data['origin_aa'].size(0)], dtype=torch.bool),
                torch.zeros([lig_max_length-data['origin_aa'].size(0)], dtype=torch.bool)
            ], dim=0)
            seq_lig_list_padded.append({'esm_embedding': esm_embedding.float(), 
                                              'origin_aa': origin_aa,
                                              'seq_mask': seq_mask, })
        
        receptor_data_batch = seq_rec_list_padded
        receptor_data_batch = default_collate(receptor_data_batch)
        receptor_data_batch['size'] = len(receptor_data_batch)
        
        ligand_data_batch = seq_lig_list_padded # [{**d1, **d2} for d1, d2 in zip(struc_lig_list_padded, seq_lig_list_padded)]
        ligand_data_batch = default_collate(ligand_data_batch)
        ligand_data_batch['size'] = len(ligand_data_batch)
        
        return {
            'receptor': receptor_data_batch,
            'ligand':ligand_data_batch
        }


class PaddingCollate_struc_infer(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n - l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):

        receptor_data_list = [d['receptor'] for d in data_list]
        ligand_data_list = [d['ligand'] for d in data_list]

        keys = {k for k in self._get_common_keys(receptor_data_list) if 'origin' not in k and 'esm' not in k}
        rec_max_length = max([data[self.length_ref_key].size(0) for data in receptor_data_list])
        lig_max_length = max([data[self.length_ref_key].size(0) for data in ligand_data_list])
        if self.eight:
            rec_max_length = math.ceil(rec_max_length / 8) * 8
            lig_max_length = math.ceil(lig_max_length / 8) * 8

        # 结构模型要用到的特征
        struc_rec_list_padded = []
        for data in receptor_data_list:
            data_padded = {
                k: self._pad_last(v, rec_max_length, value=self._get_pad_value(k))
                for k, v in data.items()
                if k in keys
            }
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), rec_max_length)
            struc_rec_list_padded.append(data_padded)

        struc_lig_list_padded = []
        for data in ligand_data_list:
            data_padded = {
                k: self._pad_last(v, lig_max_length, value=self._get_pad_value(k))
                for k, v in data.items()
                if k in keys
            }
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), lig_max_length)
            struc_lig_list_padded.append(data_padded)

        receptor_data_batch = default_collate(struc_rec_list_padded)
        receptor_data_batch['size'] = len(receptor_data_batch)

        ligand_data_batch = default_collate(struc_lig_list_padded)
        ligand_data_batch['size'] = len(ligand_data_batch)

        return {
            'receptor': receptor_data_batch,
            'ligand': ligand_data_batch
        }


class PaddingCollate_seq_infer(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n - l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        receptor_data_list = [d['receptor'] for d in data_list]
        ligand_data_list = [d['ligand'] for d in data_list]

        # 序列模型要用到的特征
        rec_max_length = max([data['origin_aa'].size(0) for data in receptor_data_list])
        lig_max_length = max([data['origin_aa'].size(0) for data in ligand_data_list])

        seq_rec_list_padded = []
        for data in receptor_data_list:
            if data['origin_aa'].size(0) < rec_max_length:
                pad_length = rec_max_length - data['origin_aa'].size(0)
                esm_embedding = F.pad(data['esm_embedding'],
                                      pad=(0, 0, 0, pad_length),
                                      mode='constant', value=0)
                origin_aa = torch.cat([
                    data['origin_aa'],
                    torch.tensor([DEFAULT_PAD_VALUES['origin_aa']] * pad_length),
                ], dim=0)
            else:
                esm_embedding = data['esm_embedding']
                origin_aa = data['origin_aa']
            seq_mask = torch.cat([
                torch.ones([data['origin_aa'].size(0)], dtype=torch.bool),
                torch.zeros([rec_max_length - data['origin_aa'].size(0)], dtype=torch.bool)
            ], dim=0)
            seq_rec_list_padded.append({'esm_embedding': esm_embedding.float(),
                                        'origin_aa': origin_aa,
                                        'seq_mask': seq_mask,
                                        'dG': data['dG'],
                                        'complex': data['complex']})

        seq_lig_list_padded = []
        for data in ligand_data_list:
            if data['origin_aa'].size(0) < lig_max_length:
                pad_length = lig_max_length - data['origin_aa'].size(0)
                esm_embedding = F.pad(data['esm_embedding'],
                                      pad=(0, 0, 0, pad_length),
                                      mode='constant', value=0)
                origin_aa = torch.cat([
                    data['origin_aa'],
                    torch.tensor([DEFAULT_PAD_VALUES['origin_aa']] * pad_length),
                ], dim=0)
            else:
                esm_embedding = data['esm_embedding']
                origin_aa = data['origin_aa']
            seq_mask = torch.cat([
                torch.ones([data['origin_aa'].size(0)], dtype=torch.bool),
                torch.zeros([lig_max_length - data['origin_aa'].size(0)], dtype=torch.bool)
            ], dim=0)
            seq_lig_list_padded.append({'esm_embedding': esm_embedding.float(),
                                        'origin_aa': origin_aa,
                                        'seq_mask': seq_mask,
                                        'dG': data['dG'],
                                        'complex': data['complex']})

        receptor_data_batch = seq_rec_list_padded
        receptor_data_batch = default_collate(receptor_data_batch)
        receptor_data_batch['size'] = len(receptor_data_batch)

        ligand_data_batch = seq_lig_list_padded  # [{**d1, **d2} for d1, d2 in zip(struc_lig_list_padded, seq_lig_list_padded)]
        ligand_data_batch = default_collate(ligand_data_batch)
        ligand_data_batch['size'] = len(ligand_data_batch)

        return {
            'receptor': receptor_data_batch,
            'ligand': ligand_data_batch
        }


class PaddingCollate_mix():
    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.eight = eight
        
        self.paddingcollate_struc = PaddingCollate_struc(length_ref_key, pad_values, eight)
        self.paddingcollate_seq = PaddingCollate_seq(length_ref_key, pad_values, eight)
    
    def __call__(self, data_list:List[Tuple]):
        data_list_struc = [data['receptor'] for data in data_list]
        data_list_seq = [data['ligand'] for data in data_list]
        batch_struc = self.paddingcollate_struc(data_list_struc)
        batch_seq = self.paddingcollate_seq(data_list_seq)
        return {'receptor':batch_struc, 'ligand':batch_seq}

