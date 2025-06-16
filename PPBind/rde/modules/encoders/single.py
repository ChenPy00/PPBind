# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from rde.modules.common.layers import AngularEncoding, RSAA_Encoding
from rde.modules.common.geometry import construct_3d_basis
from rde.utils.protein.constants import BBHeavyAtom
from rde.utils.protein.constants import AAindex1_matrix, AAindex1_matrix_dim


class PerResidueEncoder(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.aatype_embed = nn.Embedding(self.max_aa_types, feat_dim)
        self.dihed_embed = AngularEncoding()
        infeat_dim = feat_dim + self.dihed_embed.get_out_dim(6) # Phi, Psi, Chi1-4
        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, aa, phi, phi_mask, psi, psi_mask, chi, chi_mask, mask_residue):
        """
        Args:
            aa: (N, L)
            phi, phi_mask: (N, L)
            psi, psi_mask: (N, L)
            chi, chi_mask: (N, L, 4)
            mask_residue: (N, L)
        """
        N, L = aa.size()

        # Amino acid identity features
        aa_feat = self.aatype_embed(aa) # (N, L, feat)

        # Dihedral features
        dihedral = torch.cat(
            [phi[..., None], psi[..., None], chi],
            dim=-1
        ) # (N, L, 6)
        dihedral_mask = torch.cat([
            phi_mask[..., None], psi_mask[..., None], chi_mask], 
            dim=-1
        ) # (N, L, 6)
        dihedral_feat = self.dihed_embed(dihedral[..., None]) * dihedral_mask[..., None] # (N, L, 6, feat)
        dihedral_feat = dihedral_feat.reshape(N, L, -1)

        # Mix
        out_feat = self.mlp(torch.cat([aa_feat, dihedral_feat], dim=-1)) # (N, L, F)
        out_feat = out_feat * mask_residue[:, :, None]
        return out_feat


class PerResidueEncoder_aaindex1(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.aatype_embed = nn.Embedding(self.max_aa_types, feat_dim)
        self.dihed_embed = AngularEncoding()
        self.aaindex1_embed = nn.Linear(AAindex1_matrix_dim, feat_dim)
        self.rsaa_embed = RSAA_Encoding()
        infeat_dim = feat_dim + self.dihed_embed.get_out_dim(6) + self.rsaa_embed.get_out_dim(1)  # feat_dim + Phi, Psi, Chi1-4 + RSAA_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, aa, phi, phi_mask, psi, psi_mask, chi, chi_mask, mask_residue, rsaa, rsaa_mask):
        """
        Args:
            aa: (N, L)
            phi, phi_mask: (N, L)
            psi, psi_mask: (N, L)
            chi, chi_mask: (N, L, 4)
            mask_residue: (N, L)
        """
        N, L = aa.size()

        # Amino acid identity features
        aa_feat = self.aatype_embed(aa) # (N, L, feat)

        # Based on AAindex1 index features
        aaindex1_matrix = AAindex1_matrix.to(aa.device)
        aaindex1_matrix_expanded = aaindex1_matrix.unsqueeze(0).expand(aa.size(0), -1, -1)
        aa_expanded = aa.unsqueeze(-1).expand(-1, -1, aaindex1_matrix.size(-1))

        aaindex1_feat = torch.gather(aaindex1_matrix_expanded, 1, aa_expanded)  # (N, L, aaindex1_feat_dim) 
        aaindex1_feat = self.aaindex1_embed(aaindex1_feat)  # (N, L, feat) 

        aa_feat = aa_feat + aaindex1_feat
        
        # 计算rsaa feature
        rsaa_feat = self.rsaa_embed(rsaa[..., None]) * rsaa_mask[..., None]  # (N, L, 1, feat)
        rsaa_feat = rsaa_feat.reshape(N, L, -1)
        
        # Dihedral features
        dihedral = torch.cat(
            [phi[..., None], psi[..., None], chi],
            dim=-1
        ) # (N, L, 6)
        dihedral_mask = torch.cat([
            phi_mask[..., None], psi_mask[..., None], chi_mask], 
            dim=-1
        ) # (N, L, 6)
        dihedral_feat = self.dihed_embed(dihedral[..., None]) * dihedral_mask[..., None] # (N, L, 6, feat)
        dihedral_feat = dihedral_feat.reshape(N, L, -1)

        # Mix
        # aa features, Dihedral features, rsaa features
        out_feat = self.mlp(torch.cat([aa_feat, dihedral_feat, rsaa_feat], dim=-1)) # (N, L, F)
        out_feat = out_feat * mask_residue[:, :, None]
        return out_feat


class PerAtomEncoder(nn.Module):

    def __init__(self, feat_dim=8*15, max_aa_types=22):
        super().__init__()
        self.max_aa_types = max_aa_types
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim // 15 * 2 + 3, feat_dim // 15 * 2), nn.ReLU(),
            nn.Linear(feat_dim // 15 * 2, feat_dim // 15)
        )
        self.atom_embed = nn.Embedding(max_aa_types, feat_dim)
        self.atom_type_embed = nn.Parameter(torch.empty((15, feat_dim // 15)))
        nn.init.normal_(self.atom_type_embed)

    def forward(self, aa, pos14, pos14_mask):
        '''
        :param aa:          (N, L).
        :param pos14:       (N, L, 14, 3).
        :param pos14_mask:  (N, L, 14, 3).
        :return:
        '''
        N, L = aa.size()
        atom_pos = pos14.view(N, L * 15, 3)
        atom_embed = self.atom_embed(aa).reshape(N, L, 15, -1)
        atom_type_embed_expanded = self.atom_type_embed[None, None, :, :].expand(N, L, -1, -1)

        # (N, L, 14, feat_dim//14)
        out_feat = self.mlp(torch.cat([atom_type_embed_expanded,
                                       atom_embed,
                                       pos14], dim=-1))
        out_feat[~pos14_mask[..., None].expand_as(out_feat)] = 0

        c_pos = pos14[:, :, BBHeavyAtom.C, :]
        c_pos = c_pos[:, :, None, :].expand(-1, -1, 15, -1).reshape(N, L * 15, 3)

        n_pos = pos14[:, :, BBHeavyAtom.N, :]
        n_pos = n_pos[:, :, None, :].expand(-1, -1, 15, -1).reshape(N, L * 15, 3)

        R = construct_3d_basis(atom_pos, c_pos, n_pos)
        t = atom_pos
        return out_feat, R.view(N, L, 15, 3, 3), t.view(N, L, 15, 3)


