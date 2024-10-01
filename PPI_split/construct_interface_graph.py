import os
from functools import partial
from pathlib import Path
from typing import List, Dict
import warnings

from graphein.protein.config import ProteinGraphConfig
from graphein.utils.utils import annotate_node_metadata
from graphein.protein.edges.distance import add_k_nn_edges
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot
from graphein.protein.graphs import (initialise_graph_with_metadata, read_pdb_to_dataframe,
                                     sort_dataframe, process_dataframe, add_nodes_to_graph,
                                     compute_edges)


def get_interface_nodes(graph: Dict,
                        ligand_chain: List[str],
                        receptoe_chain: List[str],
                        interface_dist_threshold: float = 10):
    assert all(graph['dist_mat'].index == graph['pdb_df'].index)  # ALL must be Ture

    ligand_index = graph['pdb_df']['chain_id'].isin(ligand_chain)
    receptor_index = graph['pdb_df']['chain_id'].isin(receptoe_chain)
    dist_mat_ = graph['dist_mat'].loc[ligand_index[ligand_index].index, receptor_index[receptor_index].index]
    interface_dist_mat = dist_mat_.loc[(dist_mat_ < interface_dist_threshold).any(axis=1),
                                       (dist_mat_ < interface_dist_threshold).any(axis=0)]
    interface_ligand_index = interface_dist_mat.index
    interface_receptor_index = interface_dist_mat.columns
    interface_index = interface_ligand_index.tolist() + interface_receptor_index.tolist()
    interface_nodes = graph['pdb_df'].loc[interface_index, 'node_id'].values

    return interface_nodes, interface_index


def get_interface_graph(path,
                        ligand_chain: List[str],
                        receptoe_chain: List[str]):
    if path is not None and isinstance(path, Path):
        path = os.fsdecode(path)

    chain_selection = ligand_chain + receptoe_chain

    edge_construction_functions = [
        partial(add_k_nn_edges, k=1_000_000,
                long_interaction_threshold=0,
                exclude_edges=['inter'], kind_name='intra'),
        partial(add_k_nn_edges, k=1_000_000,
                long_interaction_threshold=0,
                exclude_edges=['intra'], kind_name='inter'),
    ]
    node_metadata_functions = [amino_acid_one_hot]

    config = ProteinGraphConfig(
        edge_construction_functions=[],
        node_metadata_functions=node_metadata_functions,
        insertions=True
    )

    raw_df = read_pdb_to_dataframe(
        path,
        model_index=1,
    )
    raw_df = sort_dataframe(raw_df)
    protein_df = process_dataframe(
        raw_df,
        chain_selection=chain_selection,
        granularity=config.granularity,
        insertions=config.insertions,
        alt_locs=config.alt_locs,
        keep_hets=config.keep_hets,
        atom_df_processing_funcs=config.protein_df_processing_functions,
        hetatom_df_processing_funcs=config.protein_df_processing_functions,
    )
    g = initialise_graph_with_metadata(
        protein_df=protein_df,
        raw_pdb_df=raw_df,
        path=path,
        granularity=config.granularity,
    )
    # Add nodes to graph
    g = add_nodes_to_graph(g)

    # Add config to graph
    g.graph["config"] = config

    # Annotate additional node metadata
    if config.node_metadata_functions is not None:
        g = annotate_node_metadata(g, config.node_metadata_functions)

    g = compute_edges(
        g,
        funcs=config.edge_construction_functions,
        get_contacts_config=None,
    )

    interface_nodes, interface_index = get_interface_nodes(g.graph, ligand_chain, receptoe_chain)
    interface_dist_mat = g.graph['dist_mat'].loc[interface_index, interface_index]
    interface_node_map = g.graph['pdb_df'][g.graph['pdb_df'].index.isin(interface_index)]['node_id']
    interface_node_map = {v: k for k, v in interface_node_map.to_dict().items()}

    # remove non interface nodes
    other_nodes = g.graph['pdb_df'][~g.graph['pdb_df']['node_id'].isin(interface_nodes)]['node_id'].values
    g.remove_nodes_from(other_nodes)

    # rebuilt edge
    for func in edge_construction_functions:
        func(g)

    for u, v, d in g.edges(data=True):
        d["distance"] = interface_dist_mat.loc[interface_node_map[u], interface_node_map[v]]

    if len(g.edges) != len(g.nodes) ** 2 / 2 - len(g.nodes) / 2:
        warnings.warn(f'{path}:The interface graph is not full edge, there are unconnected nodes')

    # update chain id
    g.graph["chain_ids"] = list(set([g.nodes[n]['chain_id'] for n in g.nodes]))

    return g
