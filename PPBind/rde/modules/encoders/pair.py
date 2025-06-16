# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from rde.modules.common.geometry import angstrom_to_nm, pairwise_dihedrals, cross_pairwise_dihedrals
from rde.modules.common.layers import AngularEncoding, RSAA_Encoding
from rde.utils.protein.constants import BBHeavyAtom


class ResiduePairEncoder(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22, max_relpos=32):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos
        self.aa_pair_embed = nn.Embedding(self.max_aa_types*self.max_aa_types, feat_dim)
        self.relpos_embed = nn.Embedding(2*max_relpos+1, feat_dim)

        self.aapair_to_distcoef = nn.Embedding(self.max_aa_types*self.max_aa_types, max_num_atoms*max_num_atoms)
        nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.distance_embed = nn.Sequential(
            nn.Linear(max_num_atoms*max_num_atoms, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
        )

        self.dihedral_embed = AngularEncoding()
        feat_dihed_dim = self.dihedral_embed.get_out_dim(2) # Phi and Psi

        infeat_dim = feat_dim+feat_dim+feat_dim+feat_dihed_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms):
        """
        Args:
            aa: (N, L).
            res_nb: (N, L).
            chain_nb: (N, L).
            pos_atoms:  (N, L, A, 3)
            mask_atoms: (N, L, A)
        Returns:
            (N, L, L, feat_dim)
        """
        N, L = aa.size()
        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA] # (N, L)
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]

        # Pair identities
        aa_pair = aa[:,:,None]*self.max_aa_types + aa[:,None,:]    # (N, L, L)
        feat_aapair = self.aa_pair_embed(aa_pair)
    
        # Relative positions
        same_chain = (chain_nb[:, :, None] == chain_nb[:, None, :])
        relpos = torch.clamp(
            res_nb[:,:,None] - res_nb[:,None,:], 
            min=-self.max_relpos, max=self.max_relpos,
        )   # (N, L, L)
        feat_relpos = self.relpos_embed(relpos + self.max_relpos) * same_chain[:,:,:,None]# ??+ self.max_relpos??

        # Distances
        d = angstrom_to_nm(torch.linalg.norm(
            pos_atoms[:,:,None,:self.max_num_atoms,None] - pos_atoms[:,None,:,None,:self.max_num_atoms],
            dim = -1, ord = 2,
        )).reshape(N, L, L, -1) # (N, L, L, A*A)
        c = F.softplus(self.aapair_to_distcoef(aa_pair))    # (N, L, L, A*A)
        d_gauss = torch.exp(-1 * c * d**2)
        mask_atom_pair = (mask_atoms[:,:,None,:self.max_num_atoms,None] * mask_atoms[:,None,:,None,:self.max_num_atoms]).reshape(N, L, L, -1)
        feat_dist = self.distance_embed(d_gauss * mask_atom_pair)

        # Orientations
        dihed = pairwise_dihedrals(pos_atoms)   # (N, L, L, 2)
        feat_dihed = self.dihedral_embed(dihed)

        # All
        feat_all = torch.cat([feat_aapair, feat_relpos, feat_dist, feat_dihed], dim=-1)
        feat_all = self.out_mlp(feat_all)   # (N, L, L, F)
        feat_all = feat_all * mask_pair[:, :, :, None]

        return feat_all


class ResiduePairEncoder_rsaa(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22, max_relpos=32):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos
        self.aa_pair_embed = nn.Embedding(self.max_aa_types*self.max_aa_types, feat_dim)
        self.relpos_embed = nn.Embedding(2*max_relpos+1, feat_dim)

        self.aapair_to_distcoef = nn.Embedding(self.max_aa_types*self.max_aa_types, max_num_atoms*max_num_atoms)
        nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.distance_embed = nn.Sequential(
            nn.Linear(max_num_atoms*max_num_atoms, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
        )

        self.dihedral_embed = AngularEncoding()
        feat_dihed_dim = self.dihedral_embed.get_out_dim(2) # Phi and Psi
        
        self.rsaa_embed = RSAA_Encoding()
        feat_rsaa_dim = self.rsaa_embed.get_out_dim(2) # Product and Difference
        
        infeat_dim = feat_dim+feat_dim+feat_dim+feat_dihed_dim+feat_rsaa_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, rsaa, rsaa_mask):
        """
        Args:
            aa: (N, L).
            res_nb: (N, L).
            chain_nb: (N, L).
            pos_atoms:  (N, L, A, 3)
            mask_atoms: (N, L, A)
        Returns:
            (N, L, L, feat_dim)
        """
        N, L = aa.size()
        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA] # (N, L)
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]
        rsaa_mask_pair = rsaa_mask[:, :, None] * rsaa_mask[:, None, :]

        # Pair identities
        aa_pair = aa[:,:,None]*self.max_aa_types + aa[:,None,:]    # (N, L, L)
        feat_aapair = self.aa_pair_embed(aa_pair)
    
        # Relative positions
        same_chain = (chain_nb[:, :, None] == chain_nb[:, None, :])
        relpos = torch.clamp(
            res_nb[:,:,None] - res_nb[:,None,:], 
            min=-self.max_relpos, max=self.max_relpos,
        )   # (N, L, L)
        feat_relpos = self.relpos_embed(relpos + self.max_relpos) * same_chain[:,:,:,None]

        # Distances
        d = angstrom_to_nm(torch.linalg.norm(
            pos_atoms[:,:,None,:self.max_num_atoms,None] - pos_atoms[:,None,:,None,:self.max_num_atoms],
            dim = -1, ord = 2,
        )).reshape(N, L, L, -1) # (N, L, L, Atom*Atom)
        d_gauss = torch.exp(-1 * d**2)

        mask_atom_pair = (mask_atoms[:,:,None,:self.max_num_atoms,None] * mask_atoms[:,None,:,None,:self.max_num_atoms]).reshape(N, L, L, -1)
        feat_dist = self.distance_embed(d_gauss * mask_atom_pair)

        # Orientations
        dihed = pairwise_dihedrals(pos_atoms)   # (N, L, L, 2)
        feat_dihed = self.dihedral_embed(dihed)
        
        # RSAA-pair
        rsaa_diff = rsaa[:, :, None] - rsaa[:,None,:]
        rsaa_prod = rsaa[:, :, None] + rsaa[:,None,:]
        rsaa_pair = torch.cat([rsaa_diff[..., None],  rsaa_prod[..., None]], dim=-1)
        feat_rsaa_pair = self.rsaa_embed(rsaa_pair[..., None])   # (N, L, L, 1, feat)
        feat_rsaa_pair = feat_rsaa_pair.reshape(N, L, L, -1) 
        feat_rsaa_pair =  feat_rsaa_pair * rsaa_mask_pair[..., None]

        # All
        feat_all = torch.cat([feat_aapair, feat_relpos, feat_dist, feat_dihed, feat_rsaa_pair], dim=-1)
        feat_all = self.out_mlp(feat_all) # (N, L, L, F)
        feat_all = feat_all * mask_pair[:, :, :, None]

        return feat_all


# +
class ResidueCrossPairEncoder_rsaa(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22, max_relpos=32):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos
        self.aa_pair_embed = nn.Embedding(self.max_aa_types*self.max_aa_types, feat_dim)
        self.relpos_embed = nn.Embedding(2*max_relpos+1, feat_dim)

        self.aapair_to_distcoef = nn.Embedding(self.max_aa_types*self.max_aa_types, max_num_atoms*max_num_atoms)
        nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.distance_embed = nn.Sequential(
            nn.Linear(max_num_atoms*max_num_atoms, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
        )

        self.dihedral_embed = AngularEncoding()
        feat_dihed_dim = self.dihedral_embed.get_out_dim(2) # Phi and Psi
        
        self.rsaa_embed = RSAA_Encoding()
        feat_rsaa_dim = self.rsaa_embed.get_out_dim(2) # Product and Difference
        
        infeat_dim = feat_dim+feat_dim+feat_dihed_dim+feat_rsaa_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, 
                aa_q, res_nb_q, chain_nb_q, pos_atoms_q, mask_atoms_q, rsaa_q, rsaa_mask_q,
                aa_kv, res_nb_kv, chain_nb_kv, pos_atoms_kv, mask_atoms_kv, rsaa_kv, rsaa_mask_kv
               ):
        """
        Args:
            aa: (N, L).
            res_nb: (N, L).
            chain_nb: (N, L).
            pos_atoms:  (N, L, A, 3)
            mask_atoms: (N, L, A)
        Returns:
            (N, L, L, feat_dim)
        """
        N, L_q = aa_q.size()
        _, L_kv = aa_kv.size()
        
        mask_residue_q = mask_atoms_q[:, :, BBHeavyAtom.CA] # (N, L_q)
        mask_residue_kv = mask_atoms_kv[:, :, BBHeavyAtom.CA] # (N, L_kv)
        mask_pair = mask_residue_q[:, :, None] * mask_residue_kv[:, None, :]
        rsaa_mask_pair = rsaa_mask_q[:, :, None] * rsaa_mask_kv[:, None, :]

        # Pair identities
        aa_pair = aa_q[:,:,None]*self.max_aa_types + aa_kv[:,None,:]# (N, L_q, L_kv)
        feat_aapair = self.aa_pair_embed(aa_pair)

        # Distances
        d = angstrom_to_nm(torch.linalg.norm(
            # 原版
            #pos_atoms[:,:,None,:,None] - pos_atoms[:,None,:,None,:],
            # LHQ改版
            pos_atoms_q[:,:,None,:self.max_num_atoms,None] - pos_atoms_kv[:,None,:,None,:self.max_num_atoms],
            dim = -1, ord = 2,
        )).reshape(N, L_q, L_kv, -1) # (N, L_q, L_kv, A*A)
        d_gauss = torch.exp(-1 * d**2)

        mask_atom_pair = (mask_atoms_q[:,:,None,:self.max_num_atoms,None] * mask_atoms_kv[:,None,:,None,:self.max_num_atoms]).reshape(N, L_q, L_kv, -1)
        feat_dist = self.distance_embed(d_gauss * mask_atom_pair)# (N, L_q, L_kv, A*A) * (N, L_q, L_kv, A*A)

        # Orientations
        dihed = cross_pairwise_dihedrals(pos_atoms_q,pos_atoms_kv)   # (N, L_q, L_kv, 2)
        feat_dihed = self.dihedral_embed(dihed)
        
        # RSAA-pair
        rsaa_diff = rsaa_q[:, :, None] - rsaa_kv[:,None,:]
        rsaa_prod = rsaa_q[:, :, None] + rsaa_kv[:,None,:]
        rsaa_pair = torch.cat([rsaa_diff[..., None],  rsaa_prod[..., None]], dim=-1)
        feat_rsaa_pair = self.rsaa_embed(rsaa_pair[..., None])   # (N, L_q, L_kv, 1, feat)
        feat_rsaa_pair = feat_rsaa_pair.reshape(N, L_q, L_kv, -1) 
        feat_rsaa_pair =  feat_rsaa_pair * rsaa_mask_pair[..., None]

        # All
        feat_all = torch.cat([feat_aapair, feat_dist, feat_dihed, feat_rsaa_pair], dim=-1)
        feat_all = self.out_mlp(feat_all) # (N, L_q, L_kv, F)
        feat_all = feat_all * mask_pair[:, :, :, None]

        return feat_all

