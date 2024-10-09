#!/usr/bin/env python
# coding: utf-8
# %%
import dill
import pandas as pd
import glob
from tqdm import tqdm


# %%
all_dill_list = glob.glob('./raw/*/*.dill')

df_dict = {
    'PDB':[],
    'l_chain':[],
    'r_chain':[],
    'l_seq':[],
    'r_seq':[],
    'pair_id':[]
}


# %%
for dill_file_path in tqdm(all_dill_list):
    with open(dill_file_path, 'rb') as f:
        data = dill.load(f)
    df_dict['PDB'].append(data[0][:4].upper())
    df_dict['l_chain'].append(data[1].chain.iloc[0])
    df_dict['r_chain'].append(data[2].chain.iloc[0])
    df_dict['l_seq'].append(data.sequences['l_b'])
    df_dict['r_seq'].append(data.sequences['r_b'])
    df_dict['pair_id'].append(data.id)


# %%
df = pd.DataFrame(df_dict)
df.to_csv('./DIPS-Plus-Pair-Data.csv')


# %%
all_pdb = set(df['PDB'].tolist())
with open('./DIPS-Plus-PDB-Set.txt', 'w')as f:
    for pdb_code in all_pdb:
        f.write(f'{pdb_code},')

