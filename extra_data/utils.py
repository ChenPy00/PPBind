# -*- coding: utf-8 -*-
# +
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Selection
from Bio.PDB import PDBParser, PDBIO
import pandas as pd
import numpy as np
import os
from Bio import PDB
from Bio.Data import IUPACData
import torch

amino_acids = IUPACData.protein_letters_3to1.keys()
amino_acids = [aa.upper() for aa in list(amino_acids)]


# +
class ChainSplitter:
    def __init__(self):
        """ Create parsing and writing objects, specify output directory. """
        self.parser = PDB.PDBParser()
        self.writer = PDB.PDBIO()

    def make_pdb(self, pdb_path, chains, out_path=None, struct=None):
        """ Create a new PDB file containing only the specified chains.

        Returns the path to the created file.

        :param pdb_path: full path to the crystal structure
        :param chain_letters: iterable of chain characters (case insensitive)
        :param overwrite: write over the output file if it exists
        """
        # Input/output files
        (pdb_dir, pdb_fn) = os.path.split(pdb_path)

        # Get structure, write new file with only given chains
        struct = self.parser.get_structure('pdb', pdb_path)
        self.writer.set_structure(struct)
        self.writer.save(out_path, select=SelectChains(chains))

        return out_path

class SelectChains(PDB.Select):
    """ Only accept the specified chains when saving. """
    def __init__(self, chain_letters):
        self.chain_letters = chain_letters

    def accept_chain(self, chain):
        return (chain.get_id() in self.chain_letters)


# +
def _index_select(v, index, n):
    if isinstance(v, torch.Tensor) and v.size(0) == n:
        return v[index]
    elif isinstance(v, list) and len(v) == n:
        return [v[i] for i in index]
    else:
        return v
    
def _index_select_data(data, index):
    return {
        k: _index_select(v, index, data['aa'].size(0))
        for k, v in data.items()
    }

def _get_CB_positions(pos_atoms, mask_atoms):
    """
    Args:
        pos_atoms:  (L, A, 3)
        mask_atoms: (L, A)
    """
    from rde.utils.protein.constants import BBHeavyAtom
    L = pos_atoms.size(0)
    pos_CA = pos_atoms[:, BBHeavyAtom.CA]   # (L, 3)
    if pos_atoms.size(1) < 5:
        return pos_CA
    pos_CB = pos_atoms[:, BBHeavyAtom.CB]
    mask_CB = mask_atoms[:, BBHeavyAtom.CB, None].expand(L, 3)
    return torch.where(mask_CB, pos_CB, pos_CA)


# -

def find_del_pos(data, pos):
    idx_ligand = torch.where(data['group_id'] == 1)[0]
    idx_receptor = torch.where(data['group_id'] == 2)[0]
    dist_pair = torch.cdist(data.pos_heavyatom[idx_ligand, 1, :], data.pos_heavyatom[idx_receptor, 1, :])  # CA原子

    idx_ligand_itf, idx_receptor_itf = torch.where(dist_pair <= 8)
    idx_ligand_itf = idx_ligand[torch.unique(idx_ligand_itf)]
    idx_receptor_itf = idx_receptor[torch.unique(idx_receptor_itf)]
    idx_itf = torch.cat([idx_ligand_itf, idx_receptor_itf])
    data['itf_flag'] = torch.full_like(data['aa'], False, dtype=torch.bool)
    data['itf_flag'][idx_itf] = True

    select_flag = (data['itf_flag'] > 0)
    data['pos_atoms'] = data['pos_heavyatom'][:, :]
    data['mask_atoms'] = data['mask_heavyatom'][:, :]
    data['bfactor_atoms'] = data['bfactor_heavyatom'][:, :]

    pos_CA = data['pos_atoms'][:, 1, :]
    pos_sel = pos_CA[select_flag]
    dist_from_sel = torch.cdist(pos_CA, pos_sel).min(dim=1)[0]
    
    ligand_idx = torch.where( data['group_id']==1 )[0]
    ligand_patch_idx = torch.argsort(dist_from_sel[ligand_idx])[:128]
    ligand_patch_idx = ligand_idx[ligand_patch_idx]
    receptor_idx = torch.where( data['group_id']==2 )[0]
    receptor_patch_idx = torch.argsort(dist_from_sel[receptor_idx])[:128]
    receptor_patch_idx = receptor_idx[receptor_patch_idx]
    patch_idx = torch.cat( (ligand_patch_idx, receptor_patch_idx), dim=0 )
    data_patch = _index_select_data(data, patch_idx)
    
    interface_pos = data_patch['resseq'][data_patch['group_id']==1][torch.argsort(data_patch['resseq'][data_patch['group_id']==1])]
    drop_pos = sorted(pos[torch.isin(pos, interface_pos)==False].tolist())
    
    return drop_pos
