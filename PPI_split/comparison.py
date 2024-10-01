import os
import io
import shutil
import warnings
import itertools
import random
import traceback
import multiprocessing
import subprocess
import concurrent.futures
from abc import ABC
from typing import Iterable, Callable, Literal, Optional
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from tqdm import tqdm
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import add_k_nn_edges
from graphein.protein.features.nodes.amino_acid import (
    amino_acid_one_hot, meiler_embedding)
from graphein.protein.features.sequence.utils import (
    aggregate_feature_over_chains, aggregate_feature_over_residues)


class PPIComparator(ABC):
    def __init__(self,
        max_workers: int = os.cpu_count() - 2,
        parallel_kind: Literal['threads', 'processes'] = 'processes',
        verbose=False
    ):
        self.max_workers = max_workers
        self.parallel_kind = parallel_kind
        self.verbose = verbose

    def compare(self, ppi0: Path, ppi1: Path) -> dict:
        raise NotImplementedError()

    def _execute_task_parallel(
        self,
        func: Callable,  
        inputs: Iterable,
        kind: str = None,
        desc: str = '',
        chunksize: int = 4  # NOTE: Seems to be crucial for executor not to freeze on > ~100K jobs
    ) -> Iterable:
        """Parallelize computation

        Args:
            func (Callable): Function to apply to each input from `inputs`
            inputs (Iterable): All inputs
            kind (str): Either 'theads' or 'process'
            desc (str): Description for the tqdm progress bar

        Returns:
            Iterable: All outputs
        """
        if kind is None:
            kind = self.parallel_kind
        if kind == 'threads':
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=32)
        elif kind == 'processes':
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=32)
        else:
            raise ValueError("Invalid 'kind'. Use 'threads' or 'processes'.")

        total_tasks = len(inputs)

        if not self.verbose:
            warnings.warn(
                'Current implementation of parallelization uses `executor.map`. Therefore tqdm '
                'progress bar only shows 0\% and 100\%.'
            )

        with tqdm(desc=f'{desc} ({executor._max_workers} {kind})', total=total_tasks) as pbar:
            results = list(executor.map(partial(self._unpacked_call, func), inputs, chunksize=chunksize))
            pbar.update(total_tasks)

        return results
    
    def _unpacked_call(self, func, args):
        return func(*args)


IDIST_EMBEDDING_KIND = Literal[
    'amino_acid_one_hot', 'esm_embedding', 'meiler_embedding'
]


class IDist(PPIComparator):
    MAX_INTERFACE_SIZE = 1_000_000

    def __init__(
        self,
        kind: IDIST_EMBEDDING_KIND = 'amino_acid_one_hot',
        near_duplicate_threshold: float = 0.04,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Prepare graph features construction
        self.kind = kind
        if kind == 'amino_acid_one_hot':
            dimer_node_metadata_functions = []
            ppi_node_metadata_functions = [amino_acid_one_hot]
        elif kind == 'meiler_embedding':
            dimer_node_metadata_functions = []
            ppi_node_metadata_functions = [meiler_embedding]
        elif kind == 'esm_embedding':
            dimer_node_metadata_functions = []
            ppi_node_metadata_functions = []
        else:
            raise ValueError('Unknown `kind` value.')
        
        self.near_duplicate_threshold = near_duplicate_threshold

        self.graphein_dimer_config = ProteinGraphConfig(
            edge_construction_functions=[],
            node_metadata_functions=dimer_node_metadata_functions,
            insertions=True,
        )
        self.graphein_ppi_config = ProteinGraphConfig(
            edge_construction_functions=[
                partial(add_k_nn_edges, k=self.MAX_INTERFACE_SIZE,
                        long_interaction_threshold=0,
                        exclude_edges=['inter'], kind_name='intra'),
                partial(add_k_nn_edges, k=self.MAX_INTERFACE_SIZE,
                        long_interaction_threshold=0,
                        exclude_edges=['intra'], kind_name='inter')
            ],
            node_metadata_functions=ppi_node_metadata_functions,
            insertions=True
        )

        self.embeddings = dict()
        self.neigh = None

    def compare(
        self,
        path0: Path,
        path1: Path
    ) -> dict:
        path0, path1 = Path(path0), Path(path1)
        pdb0, pdb1 = path0.name, path1.name

        # Encode and compare
        emb0 = self.embed(path0)
        emb1 = self.embed(path1)
        metrics = {
            'L2': np.linalg.norm(emb0 - emb1),
            'L1': np.linalg.norm(emb0 - emb1, ord=1),
            'Cosine Similarity':
                np.dot(emb0, emb1) / (np.linalg.norm(emb0)*np.linalg.norm(emb1))
        }

        # Return result dict
        return {'PPI0': pdb0, 'PPI1': pdb1} | metrics

    # TODO Accelerate with sklearn pairwise
    def compare_all_against_all(
        self,
        ppis0: Iterable[Path],
        ppis1: Iterable[Path],
        embed: bool = True
    ) -> pd.DataFrame:
        # Embed PPIs
        if embed:
            ppis = set(ppis0) | set(ppis1)
            self.embed_parallel(ppis)

        # Compare all PPIs from first set against all from second set
        pairs_to_compare = itertools.product(ppis0, ppis1)
        df = [self.compare(*x) for x in pairs_to_compare]
        df = pd.DataFrame(df)
        return df

    def embed(self, ppi_id, ppi_path) -> np.array:
        if ppi_id in self.embeddings:
            return self.embeddings[ppi_id]
        
        pdb_path = ppi_path[0]
        ligand_chain, receptor_chain = ppi_path[1]
        chains = ligand_chain + receptor_chain
        
        # Construct PPI graph
        g_ppi = construct_graph(
            config=self.graphein_ppi_config, path=pdb_path, verbose=False, chain_selection=chains
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
            msg_inter = np.mean(msg_inter, axis=0)
            msg_intra = np.mean(msg_intra, axis=0)
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

    def embed_without_exception(self, *inputs) -> np.array:
        try:
            embedding = self.embed(*inputs)
        except Exception as exc:
            print(f'{inputs[0]} led to an exception {exc}:')
            print(traceback.format_exc(), end='\n\n')
            embedding = np.full(1024, np.nan)
        return embedding

    def embed_parallel(self, ppi_dict) -> None:
        # Adapt dict for multi-processing
        if self.parallel_kind == 'processes':
            self.embeddings = multiprocessing.Manager().dict()

        # Embed in parallel
        inputs = list(ppi_dict.items())
        self._execute_task_parallel(
            self.embed_without_exception, inputs, desc='Embedding PPIs'
        )

        # Return dict back to ordinary
        self.embeddings = dict(self.embeddings)

    def deduplicate_embeddings(self) -> None:
        df_emb = self.get_embeddings()
        pad_val = -1

        # Process adjacency chunk and return duplicated ids
        def reduce_func(chunk, start):
            chunk = chunk < self.near_duplicate_threshold
            chunk &= ~np.tri(*chunk.shape, k=start).astype(bool)
            idx = chunk.sum(axis=1).nonzero()[0]
            idx += start
            idx = np.pad(idx, (0, len(chunk) - len(idx)), constant_values=pad_val)
            return idx
    
        # Iterate over chunks of adjacency matrix
        def get_chunks():
            chunks = sklearn.metrics.pairwise_distances_chunked(
                df_emb,
                n_jobs=self.max_workers,
                working_memory=sklearn.get_config()['working_memory'],
                reduce_func=reduce_func
            )
            return chunks
        
        # Get chunk size
        chunk_size = len(next(get_chunks()))
        n_chunks = int(np.ceil(len(df_emb) / chunk_size))
        
        # Run
        idx_to_remove = []
        for chunk in tqdm(get_chunks(), total=n_chunks, desc='Processing adjacency chunks'):
            chunk = chunk[chunk != pad_val]
            idx_to_remove.extend(chunk)
        names_to_remove = df_emb.index[idx_to_remove]

        # Convert to original dict format
        self.embeddings = {
            name: z for name, z in self.embeddings.items() if name not in names_to_remove
        }

    def build_index(self) -> None:
        self.neigh = sklearn.neighbors.NearestNeighbors(radius=self.near_duplicate_threshold)
        self.neigh.fit(self.get_embeddings())

    def query(self, q: np.array):
        if self.neigh is None:
            self.build_index()
        neigh_dist, neigh_ind = self.neigh.radius_neighbors(np.expand_dims(q, 0), sort_results=True)
        neigh_dist, neigh_ind = neigh_dist[0], neigh_ind[0]  # single query vector
        # TODO Optimize conversion to df
        names = self.get_embeddings().index
        neigh_ind = names[neigh_ind].to_numpy()
        return neigh_dist, neigh_ind

    def get_embeddings(self) -> pd.DataFrame:
        return pd.DataFrame(dict(self.embeddings)).T
    
    def write_embeddings(self, path: Path) -> None:
        self.get_embeddings().to_csv(path)

    def read_embeddings(self, path: Path, dropna: bool = False) -> None:
        df_idist = pd.read_csv(path, index_col=0)
        df_idist = df_idist.iloc[:, :20]
        if dropna:
            df_idist = df_idist.dropna()
        embeddings = df_idist.T.to_dict(orient='series')
        embeddings = {k: np.array(v) for k, v in embeddings.items()}
        self.embeddings = embeddings
