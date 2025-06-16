# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rde.modules.common.geometry import global_to_local, local_to_global, normalize_vector, construct_3d_basis, angstrom_to_nm
from rde.modules.common.layers import mask_zero, LayerNorm
from rde.utils.protein.constants import BBHeavyAtom


def _alpha_from_logits(logits, mask, inf=1e5):
    """
    Args:
        logits: Logit matrices, (N, L_i, L_j, num_heads).
        mask:   Masks, (N, L).
    Returns:
        alpha:  Attention weights.
    """
    N, L, _, _ = logits.size()
    mask_row = mask.view(N, L, 1, 1).expand_as(logits)  # (N, L, *, *)
    mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)  # (N, L, L, *)

    logits = torch.where(mask_pair, logits, logits - inf)
    alpha = torch.softmax(logits, dim=2)  # (N, L, L, num_heads)
    alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
    return alpha


def _cross_alpha_from_logits(logits, mask_q, mask_kv, inf=1e5):
    """
    Args:
        logits: Logit matrices, (N, L_q, L_kv, num_heads).
        mask_q:   Masks, (N, L_q).
        mask_kv:   Masks, (N, L_kv).
    Returns:
        alpha:  Attention weights.
    """
    N, L_q, L_kv, _ = logits.size()
    mask_row = mask_q[:, :, None, None].expand_as(logits)  # (N, L_q, *, *)
    mask_pair = (mask_q[:,:,None] * mask_kv[:,None,:]).view(N, L_q, L_kv, 1).expand_as(logits)  # (N, L_q, L_kv, *)

    logits = torch.where(mask_pair, logits, logits - inf)
    alpha = torch.softmax(logits, dim=2)  # (N, L, L, num_heads)
    alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
    return alpha

def _heads(x, n_heads, n_ch):
    """
    Args:
        x:  (..., num_heads * num_channels)
    Returns:
        (..., num_heads, num_channels)
    """
    s = list(x.size())[:-1] + [n_heads, n_ch]
    return x.view(*s)


class GABlock(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, value_dim=32, query_key_dim=32, num_query_points=8,
                 num_value_points=8, num_heads=12, bias=False):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.pair_feat_dim = pair_feat_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points
        self.num_heads = num_heads

        # Node
        self.proj_query = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_key = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_value = nn.Linear(node_feat_dim, value_dim * num_heads, bias=bias)

        # Pair
        self.proj_pair_bias = nn.Linear(pair_feat_dim, num_heads, bias=bias)

        # Spatial
        self.spatial_coef = nn.Parameter(torch.full([1, 1, 1, self.num_heads], fill_value=np.log(np.exp(1.) - 1.)),
                                         requires_grad=True)
        self.proj_query_point = nn.Linear(node_feat_dim, num_query_points * num_heads * 3, bias=bias)
        self.proj_key_point = nn.Linear(node_feat_dim, num_query_points * num_heads * 3, bias=bias)
        self.proj_value_point = nn.Linear(node_feat_dim, num_value_points * num_heads * 3, bias=bias)

        # Output
        self.out_transform = nn.Linear(
            in_features=(num_heads * pair_feat_dim) + (num_heads * value_dim) + (
                    num_heads * num_value_points * (3 + 3 + 1)),
            out_features=node_feat_dim,
        )

        self.layer_norm_1 = LayerNorm(node_feat_dim)
        self.mlp_transition = nn.Sequential(nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                                            nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                                            nn.Linear(node_feat_dim, node_feat_dim))
        self.layer_norm_2 = LayerNorm(node_feat_dim)

    def _node_logits(self, x):
        query_l = _heads(self.proj_query(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, qk_ch)
        key_l = _heads(self.proj_key(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, qk_ch)
        logits_node = (query_l.unsqueeze(2) * key_l.unsqueeze(1) *
                       (1 / np.sqrt(self.query_key_dim))).sum(-1)  # (N, L, L, num_heads)
        return logits_node

    def _pair_logits(self, z):
        logits_pair = self.proj_pair_bias(z)
        return logits_pair

    def _spatial_logits(self, R, t, x):
        N, L, _ = t.size()
        # Query
        query_points = _heads(self.proj_query_point(x), self.num_heads * self.num_query_points, 3)  # (N, L, n_heads * n_pnts, 3)
        query_points = local_to_global(R, t, query_points)  # Global query coordinates, (N, L, n_heads * n_pnts, 3)
        query_s = query_points.reshape(N, L, self.num_heads, -1)  # (N, L, n_heads, n_pnts*3)
        # Key
        key_points = _heads(self.proj_key_point(x), self.num_heads * self.num_query_points, 3)  # (N, L, 3, n_heads * n_pnts)
        key_points = local_to_global(R, t, key_points)  # Global key coordinates, (N, L, n_heads * n_pnts, 3)
        key_s = key_points.reshape(N, L, self.num_heads, -1)  # (N, L, n_heads, n_pnts*3)
        # Q-K Product
        sum_sq_dist = ((query_s.unsqueeze(2) - key_s.unsqueeze(1)) ** 2).sum(-1)  # (N, L, L, n_heads)
        gamma = F.softplus(self.spatial_coef)
        logits_spatial = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / (9 * self.num_query_points))) / 2)  # (N, L, L, n_heads)
        return logits_spatial
    
    def _pair_aggregation(self, alpha, z):
        N, L = z.shape[:2]
        feat_p2n = alpha.unsqueeze(-1) * z.unsqueeze(-2)  # (N, L, L, n_heads, C)
        feat_p2n = feat_p2n.sum(dim=2)  # (N, L, n_heads, C)
        return feat_p2n.reshape(N, L, -1)

    def _node_aggregation(self, alpha, x):
        N, L = x.shape[:2]
        value_l = _heads(self.proj_value(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, v_ch)
        feat_node = alpha.unsqueeze(-1) * value_l.unsqueeze(1)  # (N, L, L, n_heads, *) @ (N, *, L, n_heads, v_ch)
        feat_node = feat_node.sum(dim=2)  # (N, L, n_heads, v_ch)
        return feat_node.reshape(N, L, -1)

    def _spatial_aggregation(self, alpha, R, t, x):
        N, L, _ = t.size()
        value_points = _heads(self.proj_value_point(x), self.num_heads * self.num_value_points,
                              3)  # (N, L, n_heads * n_v_pnts, 3)
        value_points = local_to_global(R, t, value_points.reshape(N, L, self.num_heads, self.num_value_points,
                                                                  3))  # (N, L, n_heads, n_v_pnts, 3)
        aggr_points = alpha.reshape(N, L, L, self.num_heads, 1, 1) * \
                      value_points.unsqueeze(1)  # (N, *, L, n_heads, n_pnts, 3)
        aggr_points = aggr_points.sum(dim=2)  # (N, L, n_heads, n_pnts, 3)

        feat_points = global_to_local(R, t, aggr_points)  # (N, L, n_heads, n_pnts, 3)
        feat_distance = feat_points.norm(dim=-1)  # (N, L, n_heads, n_pnts)
        feat_direction = normalize_vector(feat_points, dim=-1, eps=1e-4)  # (N, L, n_heads, n_pnts, 3)

        feat_spatial = torch.cat([
            feat_points.reshape(N, L, -1),
            feat_distance.reshape(N, L, -1),
            feat_direction.reshape(N, L, -1),
        ], dim=-1)

        return feat_spatial

    def forward(self, R, t, x, z, mask):
        """
        Args:
            R:  Frame basis matrices, (N, L, 3, 3_index).
            t:  Frame external (absolute) coordinates, (N, L, 3).
            x:  Node-wise features, (N, L, F).
            z:  Pair-wise features, (N, L, L, C).
            mask:   Masks, (N, L).
        Returns:
            x': Updated node-wise features, (N, L, F).
        """
        # Attention logits
        logits_node = self._node_logits(x)# NLP自注意力
        logits_pair = self._pair_logits(z)# 简单的线性变化
        logits_spatial = self._spatial_logits(R, t, x)# 空间注意力
        # Summing logits up and apply `softmax`.
        logits_sum = logits_node + logits_pair + logits_spatial
        alpha = _alpha_from_logits(logits_sum * np.sqrt(1 / 3), mask)  # (N, L, L, n_heads)

        # Aggregate features
        feat_p2n = self._pair_aggregation(alpha, z)
        feat_node = self._node_aggregation(alpha, x)
        feat_spatial = self._spatial_aggregation(alpha, R, t, x)

        # Finally
        feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_spatial], dim=-1))  # (N, L, F)
        feat_all = mask_zero(mask.unsqueeze(-1), feat_all)
        x_updated = self.layer_norm_1(x + feat_all)
        x_updated = self.layer_norm_2(x_updated + self.mlp_transition(x_updated))
        return x_updated


class GCABlock(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, value_dim=32, query_key_dim=32, num_query_points=8,
                 num_value_points=8, num_heads=12, bias=False):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.pair_feat_dim = pair_feat_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points
        self.num_heads = num_heads

        # Node
        self.proj_query = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_key = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_value = nn.Linear(node_feat_dim, value_dim * num_heads, bias=bias)

        # Pair
        self.proj_pair_bias = nn.Linear(pair_feat_dim, num_heads, bias=bias)

        # Spatial
        self.spatial_coef = nn.Parameter(torch.full([1, 1, 1, self.num_heads], fill_value=np.log(np.exp(1.) - 1.)),
                                         requires_grad=True)
        self.proj_query_point = nn.Linear(node_feat_dim, num_query_points * num_heads * 3, bias=bias)
        self.proj_key_point = nn.Linear(node_feat_dim, num_query_points * num_heads * 3, bias=bias)
        self.proj_value_point = nn.Linear(node_feat_dim, num_value_points * num_heads * 3, bias=bias)

        # Output
        self.out_transform = nn.Linear(
            in_features=(num_heads * pair_feat_dim) + (num_heads * value_dim) + (
                    num_heads * num_value_points * (3 + 3 + 1)),
            out_features=node_feat_dim,
        )

        self.layer_norm_1 = LayerNorm(node_feat_dim)
        self.mlp_transition = nn.Sequential(nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                                            nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                                            nn.Linear(node_feat_dim, node_feat_dim))
        self.layer_norm_2 = LayerNorm(node_feat_dim)

    def _node_logits(self, x_q, x_kv):
        query_l = _heads(self.proj_query(x_q), self.num_heads, self.query_key_dim)  # (N, L, n_heads, qk_ch)
        key_l = _heads(self.proj_key(x_kv), self.num_heads, self.query_key_dim)  # (N, L, n_heads, qk_ch)
        logits_node = (query_l.unsqueeze(2) * key_l.unsqueeze(1) *
                       (1 / np.sqrt(self.query_key_dim))).sum(-1)  # (N, L, L, num_heads)
        return logits_node

    def _pair_logits(self, z):
        logits_pair = self.proj_pair_bias(z)
        return logits_pair
    
    def _spatial_logits(self, R_q, t_q, x_q, R_kv, t_kv, x_kv):
        N_q, L_q, _ = t_q.size()
        N_kv, L_kv, _ = t_kv.size()
        # Query
        query_points = _heads(self.proj_query_point(x_q), self.num_heads * self.num_query_points, 3)  # (N, L_q, n_heads * n_pnts, 3)
        query_points = local_to_global(R_q, t_q, query_points)  # Global query coordinates, (N, L_q, n_heads * n_pnts, 3)
        query_s = query_points.reshape(N_q, L_q, self.num_heads, -1)  # (N, L_q, n_heads, n_pnts*3)
        # Key
        key_points = _heads(self.proj_key_point(x_kv), self.num_heads * self.num_query_points, 3)  # (N, L, 3, n_heads * n_pnts)
        key_points = local_to_global(R_kv, t_kv, key_points)  # Global key coordinates, (N, L_kv, n_heads * n_pnts, 3)
        key_s = key_points.reshape(N_kv, L_kv, self.num_heads, -1)  # (N, L_kv, n_heads, n_pnts*3)
        # Q-K Product
        sum_sq_dist = ((query_s.unsqueeze(2) - key_s.unsqueeze(1)) ** 2).sum(-1)  # (N, L_q, L_kv, n_heads)
        gamma = F.softplus(self.spatial_coef)
        logits_spatial = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / (9 * self.num_query_points)))/ 2)  # (N, L_q, L_kv, n_heads)
        return logits_spatial
    
    def _pair_aggregation(self, alpha, z):
        N, L = z.shape[:2]
        feat_p2n = alpha.unsqueeze(-1) * z.unsqueeze(-2)  # (N, L, L, n_heads, C)
        feat_p2n = feat_p2n.sum(dim=2)  # (N, L, n_heads, C)
        return feat_p2n.reshape(N, L, -1)

    def _node_aggregation(self, alpha, x_kv):
        #N, L_kv = x_kv.shape[:2]
        N, L_q = alpha.shape[:2]
        value_l = _heads(self.proj_value(x_kv), self.num_heads, self.query_key_dim)  # (N, L_kv, n_heads, v_ch)
        feat_node = alpha.unsqueeze(-1) * value_l.unsqueeze(1)  # (N, L_q, L_kv, n_heads, *) @ (N, *, L_kv, n_heads, v_ch)
        feat_node = feat_node.sum(dim=2)  # (N, L_q, n_heads, v_ch)
        return feat_node.reshape(N, L_q, -1)

    def _spatial_aggregation(self, alpha, R_q, t_q, x_q, R_kv, t_kv, x_kv):
        N, L_q, _ = t_q.size()
        _, L_kv, _ = t_kv.size()
        value_points = _heads(self.proj_value_point(x_kv), self.num_heads * self.num_value_points,3)  # (N, L_kv, n_heads * n_v_pnts, 3)
        value_points = local_to_global(R_kv, t_kv, value_points.reshape(N, L_kv, self.num_heads, self.num_value_points,3))  # (N, L_kv, n_heads, n_v_pnts, 3)
        aggr_points = alpha.reshape(N, L_q, L_kv, self.num_heads, 1, 1) * value_points.unsqueeze(1)  # (N, *, L_kv, n_heads, n_pnts, 3)
        aggr_points = aggr_points.sum(dim=2)  # (N, L_q, n_heads, n_pnts, 3)

        feat_points = global_to_local(R_q, t_q, aggr_points)  # (N, L_q, n_heads, n_pnts, 3)
        feat_distance = feat_points.norm(dim=-1)  # (N, L_q, n_heads, n_pnts)
        feat_direction = normalize_vector(feat_points, dim=-1, eps=1e-4)  # (N, L_q, n_heads, n_pnts, 3)

        feat_spatial = torch.cat([
            feat_points.reshape(N, L_q, -1),
            feat_distance.reshape(N, L_q, -1),
            feat_direction.reshape(N, L_q, -1),
        ], dim=-1)

        return feat_spatial

    def forward(self, R_q, t_q, x_q, mask_q, R_kv, t_kv, x_kv, mask_kv, z):
        """
        Args:
            R:  Frame basis matrices, (N, L, 3, 3_index).
            t:  Frame external (absolute) coordinates, (N, L, 3).
            x:  Node-wise features, (N, L, F).
            z:  Pair-wise features, (N, L, L, C).
            mask:   Masks, (N, L).
        Returns:
            x': Updated node-wise features, (N, L, F).
        """
        # Attention logits
        logits_node = self._node_logits(x_q, x_kv)# NLP式的注意力
        logits_pair = self._pair_logits(z)# 简单的线性变化
        logits_spatial = self._spatial_logits(R_q, t_q, x_q, R_kv, t_kv, x_kv)# 空间注意力
        # Summing logits up and apply `softmax`.
        logits_sum = logits_node + logits_pair + logits_spatial# (N, L_q, L_kv, n_heads)
        alpha = _cross_alpha_from_logits(logits_sum * np.sqrt(1 / 3), mask_q, mask_kv)  # (N, L_q, L_kv, n_heads)

        # Aggregate features
        feat_p2n = self._pair_aggregation(alpha, z)
        feat_node = self._node_aggregation(alpha, x_kv)
        feat_spatial = self._spatial_aggregation(alpha, R_q, t_q, x_q, R_kv, t_kv, x_kv)

        # Finally
        feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_spatial], dim=-1))  # (N, L_q, F)
        feat_all = mask_zero(mask_q.unsqueeze(-1), feat_all)
        x_updated = self.layer_norm_1(x_q + feat_all)
        x_updated = self.layer_norm_2(x_updated + self.mlp_transition(x_updated))
        return x_updated


class GAEncoder(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, num_layers, ga_block_opt={}):
        super(GAEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            GABlock(node_feat_dim, pair_feat_dim, **ga_block_opt) 
            for _ in range(num_layers)
        ])

    def forward(self, pos_atoms, res_feat, pair_feat, mask):
        R = construct_3d_basis(
            pos_atoms[:, :, BBHeavyAtom.CA], 
            pos_atoms[:, :, BBHeavyAtom.C], 
            pos_atoms[:, :, BBHeavyAtom.N]
        )
        t = pos_atoms[:, :, BBHeavyAtom.CA]
        t = angstrom_to_nm(t)
        for block in self.blocks:
            res_feat = block(R, t, res_feat, pair_feat, mask)
        return res_feat


class GCAEncoder(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, num_layers, ga_block_opt={}):
        super(GCAEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            GCABlock(node_feat_dim, pair_feat_dim, **ga_block_opt) 
            for _ in range(num_layers)
        ])

    def forward(self, pos_atoms_q, res_feat_q, mask_q, pos_atoms_kv, res_feat_kv, mask_kv, pair_feat):
        R_q = construct_3d_basis(
            pos_atoms_q[:, :, BBHeavyAtom.CA], 
            pos_atoms_q[:, :, BBHeavyAtom.C], 
            pos_atoms_q[:, :, BBHeavyAtom.N]
        )
        t_q = pos_atoms_q[:, :, BBHeavyAtom.CA]
        t_q = angstrom_to_nm(t_q)
        
        R_kv = construct_3d_basis(
            pos_atoms_kv[:, :, BBHeavyAtom.CA], 
            pos_atoms_kv[:, :, BBHeavyAtom.C], 
            pos_atoms_kv[:, :, BBHeavyAtom.N]
        )
        t_kv = pos_atoms_kv[:, :, BBHeavyAtom.CA]
        t_kv = angstrom_to_nm(t_kv)
        
        for block in self.blocks:
            res_feat_q = block(R_q, t_q, res_feat_q, mask_q, R_kv, t_kv, res_feat_kv, mask_kv, pair_feat)# (R, t, res_feat, pair_feat, mask)
        return res_feat_q