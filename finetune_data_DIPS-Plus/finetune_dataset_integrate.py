import os
import pandas as pd
import numpy as np

# +
pretain_data_df = pd.read_csv('../benchmark_pbadAS_pbadSA_strict_20240729.csv', index_col=0)
dips_plus_df = pd.read_csv('./DIPS-Plus-Pair-Data.csv', index_col=0)

data_dict_list = []
for i,r in dips_plus_df.iterrows():
    data_dict={}
    data_dict['pdb'] = r['PDB']
    data_dict['source'] = 'DIPS-Plus'
    data_dict['mutstr'] = np.nan
    data_dict['ligand'] = r['l_chain']
    data_dict['receptor'] = r['r_chain']
    data_dict['dG'] = np.nan
    data_dict['log2er'] = np.nan
    data_dict['Nkd'] = np.nan
    data_dict['PP_ID'] = 10.0
    data_dict['Subgroup'] = np.nan
    data_dict['pdb_path'] = os.path.realpath(f'./DIPS-Plus/{r["PDB"]}.pdb')
    data_dict_list.append(data_dict)
# -

finetune_data_df = pd.DataFrame(data_dict_list)
finetune_data_df = pd.concat([pretain_data_df, finetune_data_df], ignore_index=True)
finetune_data_df.to_csv('./benchmark_pbadAS_pbadSA_dipsP_20240911.csv')
finetune_data_df.to_csv('../benchmark_pbadAS_pbadSA_dipsP_20240911.csv')

finetune_data_df.PP_ID.value_counts()
