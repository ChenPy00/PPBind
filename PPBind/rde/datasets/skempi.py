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


def get_indexes(tensor):
    unique_values, inverse_indices = torch.unique_consecutive(tensor, return_inverse=True)
    indexes = [torch.nonzero(inverse_indices == i).squeeze() for i in range(len(unique_values))]
    return indexes


def load_skempi_entries(csv_path, pdb_dir, block_list={'1KBH'}):
    df = pd.read_csv(csv_path, sep=';')
    df['dG_wt'] =  (8.314/4184)*(273.15 + 25.0) * np.log(df['Affinity_wt_parsed'])
    df['dG_mut'] =  (8.314/4184)*(273.15 + 25.0) * np.log(df['Affinity_mut_parsed'])
    # df['ddG'] = df['dG_mut'] - df['dG_wt']

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

    entries = []
    for i, row in df.iterrows():
        pdbcode, group1, group2 = row['#Pdb'].split('_')
        if pdbcode in block_list:
            continue
        mut_str = row['Mutation(s)_cleaned']
        muts = list(map(_parse_mut, row['Mutation(s)_cleaned'].split(',')))
        if muts[0]['chain'] in group1:
            group_ligand, group_receptor = group1, group2
        else:
            group_ligand, group_receptor = group2, group1

        pdb_path = os.path.join(pdb_dir, '{}.pdb'.format(pdbcode.upper()))
        if not os.path.exists(pdb_path):
            continue

        if not np.isfinite(row['dG_mut']):
            continue

        entry = {
            'id': i,
            'complex': row['#Pdb'],
            'mutstr': mut_str,
            'num_muts': len(muts),
            'pdbcode': pdbcode,
            'group_ligand': list(group_ligand),
            'group_receptor': list(group_receptor),
            'mutations': muts,
            'log2ER': None,
            'dG': np.float32(row['dG_mut']),
            'pdb_path': pdb_path,
        }
        entries.append(entry)
    
    # 添加WT
    for pdb_name, gb_df in df.groupby('#Pdb'):
        pdbcode, group1, group2 = pdb_name.split('_')
        if pdbcode in block_list:
            continue
        dG = gb_df['dG_wt'].mean(skipna=True)
        if np.isnan(dG):
            continue
        if not np.isfinite(dG):
            continue
            
        # 找出ligand和receptor,此处的mut_str是借用的，并非真实存在突变
        mut_str = gb_df['Mutation(s)_cleaned'].iloc[0]
        muts = list(map(_parse_mut, mut_str.split(',')))
        if muts[0]['chain'] in group1:
            group_ligand, group_receptor = group1, group2
        else:
            group_ligand, group_receptor = group2, group1

        pdb_path = os.path.join(pdb_dir, '{}.pdb'.format(pdbcode.upper()))
        if not os.path.exists(pdb_path):
            continue

        entry = {
            'id': i,
            'complex': pdb_name,
            'mutstr': 'None',
            'num_muts': 0,
            'pdbcode': pdbcode,
            'group_ligand': list(group_ligand),
            'group_receptor': list(group_receptor),
            'mutations': [None],
            'dG': np.float32(dG),
            'pdb_path': pdb_path,
        }
        entries.append(entry)
        i += 1
        
    # 重新排序，相同complex的相连在一起
    entries = sorted(entries, key=lambda entry: entry['complex'])
    # 重新命名id
    for i,entry in enumerate(entries): entry['id']=i

    return entries


class SkempiDataset(Dataset):

    def __init__(
        self, 
        csv_path, 
        pdb_dir, 
        cache_dir,
        cvfold_index=0, 
        num_cvfolds=3, 
        split='train', 
        split_seed=2022,
        transform=None, 
        blocklist=frozenset({'1KBH'}),
        reset=False
    ):
        super().__init__()
        self.csv_path = csv_path
        self.pdb_dir = pdb_dir
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.blocklist = blocklist
        self.transform = transform
        self.cvfold_index = cvfold_index
        self.num_cvfolds = num_cvfolds
        assert split in ('train', 'val')
        self.split = split
        self.split_seed = split_seed

        self.entries_cache = os.path.join(cache_dir, 'entries.pkl')
        self.entries = None
        self.entries_full = None
        self._load_entries(reset)

        self.structures_cache = os.path.join(cache_dir, 'structures.pkl')
        self.structures = None
        self._load_structures(reset)

    def _load_entries(self, reset):
        if not os.path.exists(self.entries_cache) or reset:
            self.entries_full = self._preprocess_entries()
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries_full = pickle.load(f)

        complex_to_entries = {}
        for e in self.entries_full:
            if e['complex'] not in complex_to_entries:
                complex_to_entries[e['complex']] = []
            complex_to_entries[e['complex']].append(e)

        complex_list = sorted(complex_to_entries.keys())
        random.Random(self.split_seed).shuffle(complex_list)

        split_size = math.ceil(len(complex_list) / self.num_cvfolds)
        complex_splits = [
            complex_list[i*split_size : (i+1)*split_size] 
            for i in range(self.num_cvfolds)
        ]

        val_split = complex_splits.pop(self.cvfold_index)
        train_split = sum(complex_splits, start=[])
        if self.split == 'val':
            complexes_this = val_split
        else:
            complexes_this = train_split

        entries = []
        for cplx in complexes_this:
            entries += complex_to_entries[cplx]
        self.entries = entries
        
    def _preprocess_entries(self):
        entries = load_skempi_entries(self.csv_path, self.pdb_dir, self.blocklist)
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
        pdbcodes = list(set([e['pdbcode'] for e in self.entries_full]))
        for pdbcode in tqdm(pdbcodes, desc='Structures'):
            parser = PDBParser(QUIET=True)
            pdb_path = os.path.join(self.pdb_dir, '{}.pdb'.format(pdbcode.upper()))
            model = parser.get_structure(None, pdb_path)[0]
            data, seq_map = parse_biopython_structure(model)
            structures[pdbcode] = (data, seq_map)
        with open(self.structures_cache, 'wb') as f:
            pickle.dump(structures, f)
        return structures

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        data, seq_map = copy.deepcopy( self.structures[entry['pdbcode']] )
        keys = {'id', 'complex', 'mutstr', 'num_muts', 'pdbcode', 'dG'}
        for k in keys:
            data[k] = entry[k]
        
        group_id = []
        for ch in data['chain_id']:
            if ch in entry['group_ligand']:
                group_id.append(1)
            elif ch in entry['group_receptor']:
                group_id.append(2)
            else:
                group_id.append(0)
        data['group_id'] = torch.LongTensor(group_id)
        
        if entry['num_muts'] > 0:
            aa_mut = data['aa'].clone()
            for mut in entry['mutations']:
                ch_rs_ic = (mut['chain'], mut['resseq'], mut['icode'])
                if ch_rs_ic not in seq_map: continue
                aa_mut[seq_map[ch_rs_ic]] = one_to_index(mut['mt'])
            data['mut_flag'] = (data['aa'] != aa_mut)
            data['aa'] = aa_mut
        else:
            data['mut_flag'] = torch.full_like(data['aa'], False, dtype=torch.bool)

        # 版本1：
        # itf_flag, 结合界面
        # ligand残基和receptor残基之间CB原子距离<8, 则认为是binding界面
        # 根据rde/utils/protein/constants.py，CB原子是4
        # 计算任意两个CB原子之间的距离
        idx_ligand = torch.where(data['group_id']==1)[0]
        idx_receptor = torch.where(data['group_id']==2)[0]
        dist_pair = torch.cdist(data.pos_heavyatom[idx_ligand,4,:], data.pos_heavyatom[idx_receptor,4,:])# 4号是CB原子
        # 找出距离小于阈值的氨基酸残基
        idx_ligand_itf, idx_receptor_itf = torch.where(dist_pair<8.0)
        idx_ligand_itf = idx_ligand[list(set(idx_ligand_itf.tolist()))].tolist()
        idx_receptor_itf = idx_receptor[list(set(idx_receptor_itf.tolist()))].tolist()
        idx_itf = sorted( idx_ligand_itf+idx_receptor_itf )
        data['itf_flag'] = torch.full_like(data['aa'], False, dtype=torch.bool)
        data['itf_flag'][idx_itf] = True


        """# 版本2：
        # 计算任意ligand group和receptor group两个氨基酸残基最近距离
        idx_itf = set()
        tmp_data_pos_heavyatom = copy.deepcopy(data.pos_heavyatom)
        tmp_data_pos_heavyatom[torch.where(data.mask_heavyatom==False)] = torch.nan# 不存在的原子，坐标赋值为nan
        idxes_ligand = torch.where(data['group_id']==1)[0].tolist()
        idxes_receptor = torch.where(data['group_id']==2)[0].tolist()
        for i,j in itertools.product(idxes_ligand,idxes_receptor):
            min_dist_pair = np.nanmin(torch.cdist(tmp_data_pos_heavyatom[i,:,:], tmp_data_pos_heavyatom[j,:,:]))
            if min_dist_pair<=5.5:
                 idx_itf = idx_itf.union(set([i,j]))
        idx_itf = sorted(list(idx_itf))
        """

        '''
        # 版本3：
        # 计算任意两条链之间的两个氨基酸残基最近距离
        chain_indexes = get_indexes(data.chain_nb)
        idx_itf = torch.tensor([])
        for i in range(len(data.chain_nb.unique())):
            for j in range(len(data.chain_nb.unique())):
                if i >= j:
                    continue
                else:
                    coords_i = data.pos_heavyatom[chain_indexes[i], 4, :]  # 4为CB原子,1为CA原子
                    coords_j = data.pos_heavyatom[chain_indexes[j], 4, :]
                    dist = torch.cdist(coords_i.view(-1, 3), coords_j.view(-1, 3))
                    mask = dist < 8

                    coords_i_interface = mask.nonzero(as_tuple=False)[:, 0]
                    coords_i_interface = torch.unique(coords_i_interface)
                    coords_i_interface += chain_indexes[i][0].item()

                    coords_j_interface = mask.nonzero(as_tuple=False)[:, 1]
                    coords_j_interface = torch.unique(coords_j_interface)
                    coords_j_interface += chain_indexes[j][0].item()
                    idx_itf = torch.cat([idx_itf, coords_i_interface, coords_j_interface], dim=0)

        idx_itf = torch.unique(idx_itf)
        data['itf_flag'] = torch.full_like(data['aa'], False, dtype=torch.bool)
        data['itf_flag'][idx_itf.int()] = True
        '''
        
        if self.transform is not None:
            try:
                data = self.transform(data)
            except IndexError:
                print(data['pdbcode'])
                print(data['itf_flag'].sum())
                import pdb
                pdb.set_trace()
        return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='./data/SKEMPI_v2/skempi_v2.csv')
    parser.add_argument('--pdb_dir', type=str, default='./data/SKEMPI_v2/PDBs')
    parser.add_argument('--cache_dir', type=str, default='./data/SKEMPI_v2_cache')
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()

    dataset = SkempiDataset(
        csv_path = args.csv_path,
        pdb_dir = args.pdb_dir,
        cache_dir = args.cache_dir,
        split = 'val',
        num_cvfolds=5,
        cvfold_index=2,
        reset=args.reset,
    )
    print(dataset[0])
    print(len(dataset))
