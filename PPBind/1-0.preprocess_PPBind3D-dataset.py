# -*- coding: utf-8 -*-
# +
import os
import copy
import random
import pickle
import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import one_to_index
import itertools

from rde.utils.protein.parsers import parse_biopython_structure
from typing import List, Dict
import esm

from rde.utils.transforms import _index_select_data
import multiprocessing
from functools import partial
import time

# +
ressymb_to_resindex = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': 20,
}
resindex_to_ressymb = dict(zip(ressymb_to_resindex.values(), ressymb_to_resindex.keys()))
    
def get_indexes(tensor):
    unique_values, inverse_indices = torch.unique_consecutive(tensor, return_inverse=True)
    indexes = [torch.nonzero(inverse_indices == i).squeeze() for i in range(len(unique_values))]
    return indexes


# -

def insert_eos(index_tensor, original_tensor, src:torch.Tensor=torch.tensor([22])):
    index_list = get_indexes(index_tensor)
    result = []
    for indexs in index_list[:-1]:
        result.append(torch.cat([original_tensor[indexs], src], dim=-1))
    result.append(original_tensor[index_list[-1]])
    return torch.cat(result)


def load_data_entries(summary_filepath, block_list={'1KBH'}):
    file_name, file_extension = os.path.splitext(summary_filepath)
    assert file_extension in ['.xlsx', '.xls', '.csv'], f"File extention of summary_filepath must in ['.xlsx', '.xls', '.csv'], but the input one is {file_extension}"
    if file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(summary_filepath)
    elif file_extension=='.csv':
        df = pd.read_csv(summary_filepath)
    df['pdb'] = df['pdb'].apply( lambda x: str(x).replace('.00E+','E') )
    df = df[~((df['dG'].isna()) & (df['Nkd'].isna()) & (df['log2er'].isna()))]
    df.reset_index(drop=True, inplace=True)
    entries = []

    def _parse_mut(mut_name):
        wt_type, mutchain, mt_type = mut_name[0], mut_name[1], mut_name[-1]
        mutseq = int(mut_name[2:-1])
        return {
            'wt': wt_type,
            'mt': mt_type,
            'chain': mutchain,
            'resseq': mutseq,
            'icode': ' ',
            'name': mut_name
        }

    for i, r in df.iterrows():
        if r['pdb'].upper() in block_list:
            continue
        try:
            entry = {
                'id': i,
                'complex': r['pdb'],
                'mutstr': 'None' if type(r['mutstr']) == float else r['mutstr'],
                'num_muts': 0 if type(r['mutstr']) == float else len(r['mutstr']),
                'pdbcode': f"{r['source']}_{r['pdb']}".upper(),
                'group_ligand': r['ligand'],
                'group_receptor': r['receptor'],
                'dimer': np.float32(len(r['ligand'])==1 and len(r['receptor'])==1), 
                'l2andr2': np.float32(len(r['ligand'])<=4 and len(r['receptor'])<=4), 
                'mutations': [None] if type(r['mutstr']) == float else list(map(_parse_mut, r['mutstr'].replace(' ','').split(','))),
                'dG': np.float32(r['dG']),
                'log2er': np.float32(r['log2er']),
                'Nkd': np.float32(r['Nkd']),
                'labels': r[['dG', 'log2er', 'Nkd']].values.astype('float32'),
                'pdb_path': r['pdb_path'],
                'PP_ID': r['PP_ID']
            }
        except:
            entry = {
                'id': i,
                'complex': r['pdb'],
                'mutstr': 'None' if type(r['mutstr']) == float else r['mutstr'],
                'num_muts': 0 if type(r['mutstr']) == float else len(r['mutstr']),
                'pdbcode': f"{r['source']}_{r['pdb']}".upper(),
                'group_ligand': r['ligand'],
                'group_receptor': r['receptor'],
                'dimer': np.float32(len(r['ligand'])==1 and len(r['receptor'])==1), 
                'l2andr2': np.float32(len(r['ligand'])<=4 and len(r['receptor'])<=4), 
                'mutations': [None] if type(r['mutstr']) == float else list(map(_parse_mut, r['mutstr'].split(','))),
                'dG': np.float32(r['dG']),
                'labels': r[['dG']].values.astype('float32'),
                'pdb_path': r['pdb_path'],
            }
        entries.append(entry)

    # 重新排序，相同complex的相连在一起
    entries = sorted(entries, key=lambda entry: entry['pdbcode'])
    # 重新命名id
    for i, entry in enumerate(entries):
        entry['id'] = i

    return entries


def get_indexes(tensor):
    unique_values, inverse_indices = torch.unique_consecutive(tensor, return_inverse=True)
    indexes = [torch.nonzero(inverse_indices == i).squeeze() for i in range(len(unique_values))]
    return indexes


# 使用ESM分别提取ligand/receptor的特征
class MixedDataset(Dataset):

    def __init__(
        self,
        summary_filepath,
        cache_dir,
        save_dir,
        blocklist=frozenset({'1KBH', '4r8i', '4R8I', '1FYT','3OGO', '5NT1'}),
        only_train_pdb=None,
        transform=None,
        reset=False,
        strict=True,
        finetune=False
    ):
        super().__init__()
        if only_train_pdb is None:
            only_train_pdb = [5,6]# ['6m17be', '7ekf', '7ekg', '7v8b', '6m0j']  # []
        self.summary_filepath = summary_filepath
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.transform = transform
        self.blocklist = blocklist
        self.only_train_pdb = only_train_pdb
        self.strict = strict
        self.finetune = finetune

        self.entries_cache = os.path.join(cache_dir, 'entries.pkl')
        self.entries = None
        self.entries_full = None
        self._load_entries(reset)

        self.structures_cache = os.path.join(cache_dir, 'structures.pkl')
        self.structures = None
        self._load_structures(reset)
        
        
        # Load ESM-2 model
        self.esm, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # esm2_t33_650M_UR50D()
        self.esm = self.esm.cuda()
        self.esm.eval()
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
        

    def _load_entries(self, reset):
        if not os.path.exists(self.entries_cache) or reset:
            self.entries_full = self._preprocess_entries()
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries_full = pickle.load(f)
        self.entries = self.entries_full

    def _preprocess_entries(self):
        entries = load_data_entries(self.summary_filepath, self.blocklist)
        with open(self.entries_cache, 'wb') as f:
            pickle.dump(entries, f)
        return entries

    def _load_structures(self, reset):
        if not os.path.exists(self.structures_cache) or reset:
            self.structures = self._preprocess_structures()
        else:
            with open(self.structures_cache, 'rb') as f:
                self.structures = pickle.load(f)

    def _preprocess_structures(self):
        structures = {}
        pass_pdb = []
        pdb_path_list = list(set([e['pdb_path'] for e in self.entries_full]))
        for pdb_path in tqdm(pdb_path_list, desc='Structures'):
            source = os.path.basename(os.path.dirname(pdb_path))
            pdbcode = os.path.splitext(os.path.basename(pdb_path))[0].upper()
            if '.ENT' in pdbcode or '.ent' in pdbcode:
                pdbcode = pdbcode[:-4]
            pdbcode = f'{source}_{pdbcode}'.upper()
                
            try:
                parser = PDBParser(QUIET=True)
                model = parser.get_structure(None, pdb_path)[0]
                data, seq_map = parse_biopython_structure(model, pdb_path)
                structures[pdbcode] = (data, seq_map)
            except Exception as e:
                # 捕获到异常时执行的代码
                print("发生错误：", e)
                print(pdbcode)
                pass_pdb.append(pdbcode)
        
        print('PASS PDB: ', pass_pdb)
        with open(self.structures_cache, 'wb') as f:
            pickle.dump(structures, f)
            
        entry = [e for e in self.entries_full if e['pdbcode'] in structures.keys()]
        if len(entry)!=len(self.entries_full):
            print([e['pdbcode'] for e in self.entries_full if e['pdbcode'] not in structures.keys()])
        with open(self.entries_cache, 'wb') as f:
            pickle.dump(self.entries_full, f)
            
        return structures
    
    def get_esm_embedding(self,data):
        index_ligand = torch.where(data['group_id']==1)[0]
        index_receptor = torch.where(data['group_id']==2)[0]
        aa_ligands = data['aa'][index_ligand]
        aa_receptors = data['aa'][index_receptor]
        ligand_chain_nb = data['chain_nb'][index_ligand]
        receptor_chain_nb = data['chain_nb'][index_receptor]

        split_aa_ligand = []
        ligand_indexes = get_indexes(ligand_chain_nb)# List[List]
        for chain_index in ligand_indexes:
            split_aa = aa_ligands[chain_index]
            split_aa_ligand.append(''.join([resindex_to_ressymb[i.item()] for i in split_aa]))

        split_aa_receptor = []
        receptor_indexes = get_indexes(receptor_chain_nb)# List[List]
        for chain_index in receptor_indexes:
            split_aa = aa_receptors[chain_index]
            split_aa_receptor.append(''.join([resindex_to_ressymb[i.item()] for i in split_aa]))

        # 添加linker
        if len(split_aa_ligand)>1:
            aa_ligand = ('G'*25).join(split_aa_ligand)
            linker_mask_ligand = torch.concat([torch.tensor([True]*len(seq)+[False]*25) for seq in split_aa_ligand],dim=0)[:-25]
        else:
            aa_ligand = split_aa_ligand[0]
            linker_mask_ligand = torch.ones(len(aa_ligand), dtype=torch.bool)

        # 添加linker
        if len(split_aa_receptor)>1:
            aa_receptor = ('G'*25).join(split_aa_receptor)
            linker_mask_receptor = torch.concat([torch.tensor([True]*len(seq)+[False]*25) for seq in split_aa_receptor],dim=0)[:-25]
        else:
            aa_receptor = split_aa_receptor[0]
            linker_mask_receptor = torch.ones(len(aa_receptor), dtype=torch.bool)

        # ligand encode
        batch_labels, _, batch_tokens = self.esm_batch_converter([(f"ligand", aa_ligand)])
        with torch.no_grad():
            results = self.esm(batch_tokens.cuda(), repr_layers=[33], return_contacts=False)
        ligand_embedding = results["representations"][33][0, 1: -1]
        ligand_embedding = ligand_embedding[linker_mask_ligand].cpu().detach()# 删除linker
        
        # receptor encode
        batch_labels, _, batch_tokens = self.esm_batch_converter([(f"receptor", aa_receptor)])
        with torch.no_grad():
            results = self.esm(batch_tokens.cuda(), repr_layers=[33], return_contacts=False)
        receptor_embedding = results["representations"][33][0, 1: -1]
        receptor_embedding = receptor_embedding[linker_mask_receptor].cpu().detach()# 删除linker
        
        L = data['aa'].size(0)
        C_emb = ligand_embedding.size(-1)
        data['esm_embedding'] = torch.zeros(L,C_emb)
        data['esm_embedding'][index_ligand[torch.concat(ligand_indexes,dim=0)]] = ligand_embedding
        data['esm_embedding'][index_receptor[torch.concat(receptor_indexes,dim=0)]] = receptor_embedding
        
        return data

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        data, seq_map = copy.deepcopy(self.structures[entry['pdbcode']])
        keys = {'id', 'complex', 'mutstr', 'num_muts', 'pdbcode', 'dG', 'log2er', 'labels', 'l2andr2', 'dimer'}
        for k in keys:
            try:
                data[k] = entry[k]
            except:
                pass
            
        data['ID'] = f"{data['pdbcode']}_{data['mutstr']}_{entry['group_receptor']}_{entry['group_ligand']}"
        
        group_id = []
        for ch in data['chain_id']:
            if ch in entry['group_ligand']:# 1表示ligand
                group_id.append(1)
            elif ch in entry['group_receptor']:# 2表示receptor
                group_id.append(2)
            else:
                group_id.append(0)
        data['group_id'] = torch.LongTensor(group_id)
        
        
        # 获得mut_flag,以及突变后的aa序列
        if entry['num_muts'] > 0:
            aa_mut = data['aa'].clone()# 此时的aa是WT的
            for mut in entry['mutations']:
                ch_rs_ic = (mut['chain'], mut['resseq'], mut['icode'])
                if ch_rs_ic not in seq_map: continue
                aa_mut[seq_map[ch_rs_ic]] = one_to_index(mut['mt'])
            data['mut_flag'] = (data['aa'] != aa_mut)
            data['aa'] = aa_mut# 此时的aa是MT的
        else:
            data['mut_flag'] = torch.full_like(data['aa'], False, dtype=torch.bool)
        
        
        # 保留receptor和ligand的链(必须在获得MT的aa和mut_flag之后再执行这一步，否则获得的MT的aa是错误的)
        idx_keep = torch.where(data['group_id']!=0)[0]
        data = _index_select_data(data,idx_keep)
        
        # 添加ESM特征
        ## 将ligand视为complex，中间用'G'*25作为linker连接，用ESM提取特征。
        # 提取的特征再按顺序投影回每个aa。
        data = self.get_esm_embedding(data)

        # labels_mask
        # data['labels_mask'] = torch.logical_not(torch.isnan(torch.from_numpy(data['labels'])))
        data['labels_mask'] = ~np.isnan(data['labels'])
        data['labels'] = np.nan_to_num(data['labels'])

        # itf_flag, 结合界面
        # ligand残基和receptor残基之间CB原子距离<8, 则认为是binding界面
        # 根据rde/utils/protein/constants.py，CB原子是4, CA原子是1
        ## 计算任意两个CB原子之间的距离
        idx_ligand = torch.where(data['group_id'] == 1)[0]
        idx_receptor = torch.where(data['group_id'] == 2)[0]
        dist_pair = torch.cdist(data['pos_heavyatom'][idx_ligand, 1, :], data['pos_heavyatom'][idx_receptor, 1, :])  # 1号是CA原子
        ## 找出距离小于阈值的氨基酸残基
        idx_ligand_itf, idx_receptor_itf = torch.where(dist_pair < 10.0)# paper1取阈值为：CA原子距离10
        idx_ligand_itf = idx_ligand[torch.unique(idx_ligand_itf)]
        idx_receptor_itf = idx_receptor[torch.unique(idx_receptor_itf)]
        idx_itf = torch.cat([idx_ligand_itf, idx_receptor_itf])
        data['itf_flag'] = torch.full_like(data['aa'], False, dtype=torch.bool)
        data['itf_flag'][idx_itf] = True
        data['has_itf'] = data['itf_flag'].sum()>0

        return data
    
    def getitem_and_save_as_pt(self,i):
        """将预处理好的样本保存为pkl文件。"""
        data = self.__getitem__(i)
        save_filepath = os.path.join(self.save_dir, f"{data['ID']}.pt")
        torch.save(data, save_filepath)
#         print(f">>> Sucessfully process and save item {i} as {save_filepath}")
        return save_filepath
        
    
    def preprocess_all(self,):
        """
        将整个数据集的样本，逐个处理后保存为pt文件
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        for i in tqdm(range(self.__len__()), desc='Preprocessing dataset...'):
            save_filepath = self.getitem_and_save_as_pt(i)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_filepath', type=str, default='../PPBind-3D_example_data.csv')
    parser.add_argument('--cache_dir', type=str, default='./cache_data/PPBind-3D_example_data/')
    parser.add_argument('--save_dir', type=str, default='./cache_data/PPBind-3D_example_data/pt/')
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args(args=[])

    dataset = MixedDataset(
        summary_filepath=args.summary_filepath,
        cache_dir=args.cache_dir,
        save_dir=args.save_dir,
        reset=args.reset,
    )
    dataset.preprocess_all()
