#!/usr/bin/env python
# coding: utf-8
# %%
import pandas as pd
import requests
import wget
import os
import urllib
from multiprocessing.pool import ThreadPool


# %%
import torch
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Selection
from Bio.PDB import PDBParser, PDBIO
from Bio import PDB
from Bio.Data import IUPACData


# %%
from utils import *


# %%
import sys
sys.path.append('../PPBind-3D/')
from rde.utils.protein.parsers import parse_biopython_structure


# %%
label_name_list = ['dG', 'log2er', 'Nkd']
# Read benchmark data (processed segmentation)
# Among them, PP-ID is the grouping group

# %%
def mutstr_transform(mutstr):
    '''
    The original mutstr composition was: C_R89P represents the 89th position on the C chain, where R becomes P
    After conversion, the composition of mutstr is: RC89P represents the transformation from R to P at position 89 on the C chain
    '''
    if type(mutstr)==str:
        mutstr=mutstr.replace('_', '').replace(" ",'')
        mutstr = ','.join([m[1]+m[0]+m[2:] for m in mutstr.split(',')])
    else:
        pass
    return mutstr


# %%
import pandas as pd
benchmark_df = pd.read_csv('../PPI_split/processed_datasplit.csv', index_col=0)
benchmark_df['mutstr'] = benchmark_df['mutstr'].apply(mutstr_transform)

for label_name in label_name_list:
    if label_name in benchmark_df.columns:
        pass
    else:
        benchmark_df[label_name] = np.full(len(benchmark_df), np.nan)
        
display(benchmark_df)

# %%
# # Processing PBAD_SA data
id2PDBcode = {
    'N501Y': '7ekf',
    'Beta': '7ekg',
    'Delta': '7v8b',
    'Wuhan-Hu-1': '6m0j'
}

IR_of_pdb = {
    '7ekf':{'group_ligand':'B', 'group_receptor':'A'},
    '7ekg':{'group_ligand':'B', 'group_receptor':'A'},
    '7v8b':{'group_ligand':'A', 'group_receptor':'F'},
    '6m0j':{'group_ligand':'E', 'group_receptor':'A'},    
}

PBAD_SA_path = os.path.abspath('./PBAD-SA/')
PBAD_SA_df = []
for key, pdb_name in id2PDBcode.items():
    parser = PDBParser(QUIET=True)
    pdb_path = os.path.join(PBAD_SA_path, f'{pdb_name}.pdb')
    model = parser.get_structure(None, pdb_path)[0]
    data, seq_map = parse_biopython_structure(model, pdb_path)
    entry = IR_of_pdb[pdb_name]

    group_id = []
    for ch in data['chain_id']:
        if ch in entry['group_ligand']:
            group_id.append(1)
        elif ch in entry['group_receptor']:
            group_id.append(2)
        else:
            group_id.append(0)
    data['group_id'] = torch.LongTensor(group_id)
    
    df = pd.read_csv('./PBAD-SA/science.abo7896_data_s1.csv', usecols=['target', 'position', 'mutation', 'bind'])
    df = df[df['target']==key]
    pos = torch.tensor(list(set(df['position'].tolist())), dtype=torch.long)
    
    drop_pos = find_del_pos(data, pos)
    df = df[~df['position'].isin(drop_pos)]
    df.reset_index(drop=True, inplace=True)
    
    for position, gb_df in df.groupby('position'):
        for i, r in gb_df.iterrows():
                mutation = r['mutation']
                PBAD_SA_df.append({'pdb': pdb_name, 
                                   'mutstr': mutation[0]+entry['group_ligand'] + mutation[1:],
                                   'ligand': entry['group_ligand'], 'receptor': entry['group_receptor'], 
                                   'Nkd': r['bind'],'source':'pbad-sa', 'pdb_path': pdb_path})

PBAD_SA_df = pd.DataFrame(PBAD_SA_df)
PBAD_SA_df = PBAD_SA_df[~PBAD_SA_df['Nkd'].isna()]
PBAD_SA_df['PP_ID'] = [5]*len(PBAD_SA_df)

for label_name in label_name_list:
    if label_name in PBAD_SA_df.columns:
        pass
    else:
        PBAD_SA_df[label_name] = np.full(len(PBAD_SA_df), np.nan)

# %%
# # Processing PBAD_AS data
PBAD_AS_df = []
PBAD_AS_path = os.path.abspath('./PBAD-AS/6m17be.pdb')

parser = PDBParser(QUIET=True)
model = parser.get_structure(None, PBAD_AS_path)[0]
data, seq_map = parse_biopython_structure(model, PBAD_AS_path)
entry = {'group_ligand':'B', 'group_receptor':'E'}

group_id = []
for ch in data['chain_id']:
    if ch in entry['group_ligand']:
        group_id.append(1)
    elif ch in entry['group_receptor']:
        group_id.append(2)
    else:
        group_id.append(0)
data['group_id'] = torch.LongTensor(group_id)

df = pd.read_excel('./PBAD-AS/abc0870_data_file_s1.xlsx', header=7, usecols=['Residue #'])
pos = list(set(df[~df['Residue #'].isna()]['Residue #'].tolist()))
pos = sorted([int(p) for p in pos for _ in range(20)])

df = pd.read_excel('./PBAD-AS/abc0870_data_file_s1.xlsx', header=7,nrows=2340, 
                   usecols=['Substitution', 'Reads in Naive Library', 'Replicate 1', 'Replicate 2'])
df['pos']=pos
df['log2er']=df[['Replicate 1', 'Replicate 2']].mean(axis=1)

pos = torch.tensor(list(set(pos)), dtype=torch.long)
drop_pos = find_del_pos(data, pos)
df = df[~df['pos'].isin(drop_pos)]
df.reset_index(drop=True, inplace=True)

for position, gb_df in df.groupby('pos'):
    wt_aa = gb_df[gb_df['Reads in Naive Library']=='WT']['Substitution'].values[0].upper()
    for i, r in gb_df.iterrows():
        if r['Reads in Naive Library']=='WT':
            continue
        else:
            mut_aa = r['Substitution']
            PBAD_AS_df.append({'pdb': '6m17be', 
                               'mutstr': f'{wt_aa}B{position}{mut_aa}',
                               'ligand': 'B', 'receptor': 'E', 
                               'log2er': r['log2er'],
                               'source':'pbad-as', 
                               'pdb_path': PBAD_AS_path})

PBAD_AS_df = pd.DataFrame(PBAD_AS_df)
for label_name in label_name_list:
    if label_name in PBAD_AS_df.columns:
        pass
    else:
        PBAD_AS_df[label_name] = np.full(len(PBAD_AS_df), np.nan)

PBAD_AS_df['PP_ID'] = [6]*len(PBAD_AS_df)


# %%
all_data_df = pd.concat([benchmark_df, PBAD_SA_df, PBAD_AS_df], ignore_index=True)
print(all_data_df.shape)
all_data_df = all_data_df.drop_duplicates(subset=['pdb', 'mutstr', 'source', 'ligand', 'receptor'], ignore_index=True)
print(all_data_df.shape)
assert len(all_data_df[~all_data_df['pdb_path'].apply(lambda x: os.path.exists(x))])==0
assert len(set(all_data_df['pdb'].tolist())) == len(set([pdb.upper() for pdb in all_data_df['pdb'].tolist()]))

all_data_df = all_data_df[~( (all_data_df['PP_ID'].isin([0,1,2,3,4])) & (all_data_df['dG'].isna()) )]
all_data_df.reset_index(drop=True, inplace=True)
print(all_data_df.shape)
all_data_df.to_csv('../benchmark_pbadAS_pbadSA_strict.csv')

