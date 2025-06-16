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

if __name__ == "__main__":
    import sys
    sys.path.append('../../')
    
from rde.utils.protein.parsers import parse_biopython_structure
from typing import List, Dict
import copy

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


def load_data_entries(summary_filepath, blocklist=[]):
    file_name, file_extension = os.path.splitext(summary_filepath)
    assert file_extension in ['.xlsx', '.xls', 'csv'], f"File extention of summary_filepath must in ['.xlsx', '.xls', 'csv'], but the input one is {file_extension}"
    if file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(summary_filepath)
    elif file_extension=='.csv':
        df = pd.read_csv(summary_filepath)
        
    df = pd.read_csv(summary_filepath)
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
        if r['pdb'].upper() in blocklist:
            import pdb; pdb.set_trace()
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


# +
class MixedDataset(Dataset):

    def __init__(
        self,
        summary_filepath,
        pkl_cache_dir,
        cache_dir,
        cvfold_index=0,
        num_cvfolds=5,
        split='train',
        split_seed=2024,
        label_type=['dG','log2er','Nkd'],
        blocklist=[],
        only_train_pdb=None,
        transform=None,
        reset=False,
        strict=True,
        finetune=False,
        l2andr2=False,
    ):
        super().__init__()
        if only_train_pdb is None:
            only_train_pdb = [5,6,10]# ['6m17be', '7ekf', '7ekg', '7v8b', '6m0j']  # []
        self.summary_filepath = summary_filepath
        self.pkl_cache_dir = pkl_cache_dir
        os.makedirs(pkl_cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.transform = transform
        self.blocklist = blocklist
        self.only_train_pdb = only_train_pdb
        self.cvfold_index = cvfold_index
        self.num_cvfolds = num_cvfolds
        assert split in ('train', 'val')
        self.split = split
        self.strict = strict
        self.finetune = finetune
        self.split_seed = split_seed

        self.label_type = label_type
        self.label_type_2_index = {'dG':0,'log2er':1,'Nkd':2}
        
        self.l2andr2 = l2andr2
        
        self.entries_cache = os.path.join(pkl_cache_dir, 'entries.pkl')
        self.entries = None
        self.entries_full = None
        self._load_entries(reset)

        self.structures_cache = os.path.join(pkl_cache_dir, 'structures.pkl')
        self.structures = None
#         self._load_structures(reset)

    def _load_entries(self, reset):
        if not os.path.exists(self.entries_cache) or reset:
            self.entries_full = self._preprocess_entries()
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries_full = pickle.load(f)
            self.entries_full = [e for e in self.entries_full if e['complex'] not in self.blocklist]
        
        if 'has_itf' in self.entries_full[0]:
            self.entries_full = [e for e in self.entries_full if e['has_itf']]
        
        # Filter r2l2 (receptor no more than 2 chains, ligand no more than 2 chains)
        if self.l2andr2:
            self.entries_full = [e for e in self.entries_full if e['l2andr2']]

        # Filter target label types
        label_index = [self.label_type_2_index[i] for i in self.label_type]
        index_keep = []
        
        if self.finetune:
            pass
        else:
            for i,entry in enumerate(self.entries_full):
                if not np.isnan(entry['labels'][label_index]).all():
                    index_keep.append(i)
            self.entries_full = [self.entries_full[i] for i in index_keep]

        # Separate dG samples and non-dG samples (only for training)
        index_cv = []
        index_only_train = []
        for i,entry in enumerate(self.entries_full):
            if np.isnan(entry['labels'][self.label_type_2_index['dG']]):
                index_only_train.append(i) 
            else:
                index_cv.append(i)  
        self.entries_cv = [self.entries_full[i] for i in index_cv] 
        self.entries_only_train = [self.entries_full[i] for i in index_only_train] 

        # Strict data splitting
        if self.strict:
            complex_to_entries = {}
            for e in self.entries_full:
                # set([e['PP_ID'] for e in self.entries_full]) == {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
                if e['PP_ID'] not in complex_to_entries:
                    complex_to_entries[e['PP_ID']] = []
                complex_to_entries[e['PP_ID']].append(e)

            complex_list = sorted(complex_to_entries.keys())
            random.Random(self.split_seed).shuffle(complex_list)

            # pop only-train-pdb
            complex_list = [complex for complex in complex_list if complex not in self.only_train_pdb]
            
            random.Random(self.split_seed).shuffle(complex_list)

            split_size = math.ceil(len(complex_list) / self.num_cvfolds)
            complex_splits = [
                complex_list[i * split_size: (i + 1) * split_size]
                for i in range(self.num_cvfolds)
            ]

            val_split = complex_splits.pop(self.cvfold_index)
            train_split = sum(complex_splits, start=[])

#             # set only-train-pdb to train_split
#             if len(self.only_train_pdb) != 0:
#                 train_split += self.only_train_pdb
#                 random.Random(self.split_seed).shuffle(train_split)

            if self.split == 'val':
                complexes_this = val_split
            else:
                complexes_this = train_split

            entries = []
            for cplx in complexes_this:
                entries += complex_to_entries[cplx]
             
            # add not-dG samples to training set
            if self.split == 'train':
                entries = entries + copy.deepcopy(self.entries_only_train)
                random.Random(self.split_seed).shuffle(entries)
            
            self.entries = entries
        
        # Non-strict data splitting
        else:
            complex_to_entries = {}
            for e in self.entries_full:
                if e['PP_ID'] not in complex_to_entries:
                    complex_to_entries[e['PP_ID']] = []
                complex_to_entries[e['PP_ID']].append(e)
            
            all_complex_list = sorted(complex_to_entries.keys())
#             all_complex_list = [complex for complex in all_complex_list if complex not in self.pass_pdb]
            
            # pop only-train-pdb
            complex_list = [complex for complex in all_complex_list if complex not in self.only_train_pdb]
            only_train_complex_list = [complex for complex in all_complex_list if complex in self.only_train_pdb]
            
            entries = []  # All dG samples
            for cplx in complex_list:
                entries += complex_to_entries[cplx]
            random.Random(self.split_seed).shuffle(entries)
            
            only_train_entries = []  # All non-dG samples
            for cplx in only_train_complex_list:
                only_train_entries += complex_to_entries[cplx]
            random.Random(self.split_seed).shuffle(only_train_entries)
                        
            if self.num_cvfolds>1:
                split_size = math.ceil(len(entries) / self.num_cvfolds)
                complex_splits = [
                    entries[i * split_size: (i + 1) * split_size]
                    for i in range(self.num_cvfolds)
                ]
                val_split = complex_splits.pop(self.cvfold_index)
                train_split = sum(complex_splits, start=[])
            else:
                split_size = math.ceil(len(entries) / 5)
                val_split = entries[:split_size]
                train_split = entries[split_size:]
        
#             set only-train-pdb to train_split
            if len(self.only_train_pdb) != 0:
                train_split += only_train_entries
                random.Random(self.split_seed).shuffle(train_split)

            if self.split == 'val':
                entries = val_split
            else:
                entries = train_split

            # Filter r2l2 (receptor no more than 2 chains, ligand no more than 2 chains)
            if self.l2andr2:
                entries = [e for e in entries if e['l2andr2']]
            
            self.entries = entries

        # 添加ID & 调整label
        for entry in self.entries: 
            entry['ID'] = f"{entry['pdbcode']}_{entry['mutstr']}_{entry['group_receptor']}_{entry['group_ligand']}"


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
            except:
                print(pdbcode)
                pass_pdb.append(pdbcode)
        
        print('PASS PDB: ', pass_pdb)
        with open(self.structures_cache, 'wb') as f:
            pickle.dump(structures, f)
            
        entry = [e for e in self.entries_full if e['pdbcode'] in structures.keys()]
        if len(entry)!=len(self.entries_full):
            import pdb
            pdb.set_trace()
        with open(self.entries_cache, 'wb') as f:
            pickle.dump(self.entries_full, f)
            
        return structures

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        
        cache_file = os.path.join(self.cache_dir,f"{entry['ID']}.pt")
        assert os.path.exists(cache_file), f"Error: file not found {cache_file}"
        data = torch.load(cache_file)

        label_index = [self.label_type_2_index[i] for i in self.label_type]
        data['labels'] = data['labels'][label_index]
        data['labels_mask'] = data['labels_mask'][label_index]
        data.pop('has_itf')
        # transform
        if self.transform is not None:
            data = self.transform(data)
        return data