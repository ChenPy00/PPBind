# -*- coding: utf-8 -*-
# +
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import traceback
import multiprocessing
from matplotlib import pyplot as plt
import pymetis as metis
from functools import cmp_to_key
import networkx as nx

from graphein.protein.features.sequence.utils import (
    aggregate_feature_over_chains, aggregate_feature_over_residues)

from comparison import IDist
from construct_interface_graph import get_interface_graph

# +
path_dict = {
    'SKEMPI v2.0': '../benchmark_data/PDB/SKEMPI v2.0/',
    'PDBbind v2020': '../benchmark_data/PDB/PDBbind v2020/',
    'SAbDab': '../benchmark_data/PDB/SAbDab/',
    'ATLAS': '../benchmark_data/PDB/ATLAS/',
    'Affinity Benchmark v5.5': '../benchmark_data/PDB/Affinity Benchmark v5.5/',
}

case_type = {
    'SKEMPI v2.0': 'upper.pdb',
    'PDBbind v2020': 'lower.ent.pdb',
    'SAbDab': 'lower.pdb',
    'ATLAS': 'upper.pdb',
    'Affinity Benchmark v5.5': 'upper.pdb',
}


# -

def create_pdb_path(inputs):
    pdb_code, source = inputs
    c_t = case_type[source]
    c_t = c_t.split('.')
    suffix = '.'+'.'.join(c_t[1:])
    case = c_t[0]
    file_name = eval(f'pdb_code.{case}()')+suffix
    pdb_path = os.path.join(path_dict[source], file_name)
    if os.path.exists(pdb_path):
        return os.path.realpath(pdb_path)
    else:
        import pdb
        pdb.set_trace()


def get_csr_data(G:nx.Graph):
    edges = [(i, j, G.adj[i][j]['weight']) for i,j in G.edges()]
    edgesList = edges + [(j, i, w) for i, j, w in edges]
    cmp = lambda t1,t2 : t1[1]-t2[1] if t1[0]==t2[0] else t1[0]-t2[0]
    edgesList.sort(key = cmp_to_key(cmp)) 
    adjncy = [j for i,j,w in edgesList]
    eweights = [w for i,j,w in edgesList]
    xadj, xid = [0, ], 0 
    if not edgesList is []:
        for i in range(1, len(edgesList)):
            while xid < edgesList[i][0]:
                xadj.append(i)
                xid += 1
        xadj.append(len(edgesList)) 
    return (xadj, adjncy, eweights)


class Interface_Dist(IDist):
    def __init__(self, parallel_kind, near_duplicate_threshold):
        super().__init__(**{'parallel_kind':parallel_kind, 'near_duplicate_threshold':near_duplicate_threshold})
        
    def interface_embed(self, ppi_id, ppi_path) -> np.array:
        if ppi_id in self.embeddings:
            return self.embeddings[ppi_id]
        
        pdb_path = ppi_path[0]
        ligand_chain, receptor_chain = ppi_path[1]
        chains = ligand_chain + receptor_chain
        
        # Construct PPI graph
        g_ppi = get_interface_graph(
            path=pdb_path, ligand_chain=ligand_chain, receptoe_chain=receptor_chain
        )

        # Note: Can be significantly accelerated via graph-level matmuls
        # Aggregate neighborhoods
        for v in g_ppi.nodes():
            msg_inter = []
            msg_intra = []
            for n, e in g_ppi[v].items():
                signal = \
                    np.exp(-(e['distance']/4)**2) * g_ppi.nodes[n][self.kind]
                if 'inter' in e['kind']:
                    msg_inter.append(-signal)
                elif 'intra' in e['kind']:
                    msg_intra.append(signal)
            msg_inter = np.mean(msg_inter, axis=0) if len(msg_inter)>0 else 0
            msg_intra = np.mean(msg_intra, axis=0) if len(msg_intra)>0 else 0
            
            msg = np.mean([msg_inter, msg_intra], axis=0)
            g_ppi.nodes[v]['embedding'] = np.mean([
                g_ppi.nodes[v][self.kind],
                msg
            ], axis=0)

        # Aggregate residues in chains and then chain embeddings
        aggregate_feature_over_residues(g_ppi, 'embedding', 'mean')
        aggregate_feature_over_chains(g_ppi, 'embedding_mean', 'mean')

        # Save to cache and return
        embedding = g_ppi.graph['embedding_mean_mean']
        self.embeddings[ppi_id] = embedding
        return embedding
    
    def interface_embed_without_exception(self, *inputs) -> np.array:
        try:
            embedding = self.interface_embed(*inputs)
        except Exception as exc:
            print(f'{inputs[0]} led to an exception {exc}:')
            print(traceback.format_exc(), end='\n\n')
            embedding = np.full(1024, np.nan)
        return embedding
    
    def interface_embed_parallel(self, ppi_dict) -> None:
        # Adapt dict for multi-processing
        if self.parallel_kind == 'processes':
            self.embeddings = multiprocessing.Manager().dict()

        # Embed in parallel
        inputs = list(ppi_dict.items())
        self._execute_task_parallel(
            self.interface_embed_without_exception, inputs, desc='Embedding PPIs'
        )

        # Return dict back to ordinary
        self.embeddings = dict(self.embeddings)


# +
affinity_data = pd.read_excel('../benchmark_data/PPB-Affinity.xlsx', 
                              usecols=['PDB', 'Source Data Set', 'Mutations', 'Ligand Chains', 'Receptor Chains', 'Subgroup', 'KD(M)'],
                             dtype={"PDB":str}
                             )
affinity_data.rename(columns={'Source Data Set': 'source',
                              'Mutations':'mutstr', 'PDB':'pdb',
                              'Ligand Chains': 'ligand', 'Receptor Chains': 'receptor'}, inplace=True)

print(affinity_data.shape)
affinity_data['dG'] = (8.314/4184)*(273.15 + 25.0) * np.log(affinity_data['KD(M)'])
affinity_data.reset_index(drop=True, inplace=True)
affinity_data['pdb_path'] = affinity_data[['pdb', 'source']].apply(create_pdb_path, axis=1)
print(affinity_data.shape)
TCR_list = set(affinity_data[affinity_data.Subgroup=='TCR-pMHC']['pdb'].tolist())
AB_list = set(affinity_data[affinity_data.Subgroup=='Antibody-Antigen']['pdb'].tolist())
# -

# pdb_source : pdb_path
gdb_df = affinity_data.groupby(by=['pdb', 'source', 'pdb_path'])
ppi_dict = {}
for (pdb_code, source, pdb_path), df_ in gdb_df:
    ligand_chain = df_['ligand'].iloc[0].replace(' ','').split(',')
    receptor_chain = df_['receptor'].iloc[0].replace(' ','').split(',')
    pdb_path = os.path.realpath(pdb_path)
    ppi_dict[f'{pdb_code}_{source}'] = [pdb_path, [ligand_chain, receptor_chain]]

# +
# # compute embedding
interface_dist = Interface_Dist(parallel_kind='processes', near_duplicate_threshold=0.05)
wrong_log = {}
for inputs in tqdm(list(ppi_dict.items())):
    try:
        embedding = interface_dist.interface_embed(*inputs)
    except:
        wrong_log[inputs[0]] = inputs[1]
        continue

embedding_df = interface_dist.get_embeddings()
interface_dist.build_index()
embedding_df[['pdb_path', 'chains']]=[ppi_dict[x] for x in embedding_df.index]
embedding_df.to_csv('./pdb_interface_embedding.csv')

import json
print(wrong_log)
with open("./wrong_log_pdb.json","w", encoding='utf-8') as f:
    f.write(  json.dumps(   wrong_log  ,ensure_ascii=False, indent=4  )  )  
# -

# load embedding
interface_dist = Interface_Dist(parallel_kind='processes', near_duplicate_threshold=0.05)
embeddings = pd.read_csv('./pdb_interface_embedding.csv', index_col=0)
print(embeddings.shape)
embeddings = embeddings[~embeddings.isna().any(axis=1)]
print(embeddings.shape)
embeddings_ = embeddings.T.to_dict(orient='series')
embeddings_ = {k: np.array(v[:20]) for k, v in embeddings_.items()}
interface_dist.embeddings = embeddings_
interface_dist.build_index()

color_map = []
for key in interface_dist.embeddings.keys():
    pdb_code = key[:4]
    if pdb_code in TCR_list:
        color_map.append('red')
    elif pdb_code in AB_list:
        color_map.append('blue')
    else:
        color_map.append('#C1CDCD')

# +
G = nx.from_scipy_sparse_matrix(interface_dist.neigh.radius_neighbors_graph())
# nx.draw(G, with_labels=True)
fig, ax = plt.subplots(1,1, figsize=(10,10))
nx.draw(G,ax=ax,node_size=15,alpha=0.7, width=0.4, pos=nx.drawing.spring_layout(G, 0.07), node_color=color_map)
plt.savefig('./data_partition_result_spring_layout.tiff',dpi=350)

(xadj, adjncy, eweights) = get_csr_data(G)
eweights = np.array(eweights, dtype=int)

part_num = 5
n_cuts, membership = metis.part_graph(nparts=part_num, xadj=xadj, adjncy=adjncy, eweights=eweights)

nodes_parts = []
for p in range(part_num):
    nodes_parts.append(np.argwhere(np.array(membership) == p).ravel())
# -

assert len(G)==len(ppi_dict)

for pdb_key, ppi_group in zip(list(ppi_dict.keys()), membership):
    pdb, source = pdb_key.split('_')
    affinity_data.loc[(affinity_data['pdb']==pdb)&(affinity_data['source']==source), 'PP_ID'] = ppi_group

affinity_data.to_csv('./processed_datasplit.csv')
affinity_data.to_csv('../processed_datasplit.csv')
