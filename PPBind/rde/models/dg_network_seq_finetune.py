# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# +
from rde.modules.encoders.single import PerResidueEncoder, PerAtomEncoder, PerResidueEncoder_aaindex1
from rde.modules.encoders.pair import ResiduePairEncoder, ResiduePairEncoder_rsaa
from rde.modules.encoders.attn import GAEncoder, GAEncoderAtom
from rde.modules.mono.monotonic import MonoRegularLayer
from rde.utils.protein.constants import BBHeavyAtom
from rde.modules.common.layers import LayerNorm

from .rde import CircularSplineRotamerDensityEstimator
from .dg_network_str import  NetworkStr
from rde.utils.protein.constants import BBHeavyAtom, get_aaindex1#AAindex1_matrix, AAindex1_matrix_dim
from rde.modules.encoders.seq import *

AAindex1_matrix = get_aaindex1(lowrank=False)
AAindex1_matrix_dim = AAindex1_matrix.size(1)


# +
class NetworkSeqAlign(nn.Module):

    def __init__(self, cfg, str_cfg=None):
        super().__init__()
        self.cfg = cfg
        self.align = self.cfg.align if 'align' in self.cfg else False
        
        if self.align:
            assert str_cfg!=None, "if align=True, str_cfg must not be None"
            self.struc_model = NetworkStr(str_cfg)
            for name, param in self.struc_model.named_parameters():
                param.requires_grad = False
            for name, param in self.struc_model.predictor.named_parameters():
                param.requires_grad = True
        
        # encoder
        self.aa_embedding = nn.Embedding(22, cfg.encoder.node_feat_dim)
        self.seq_fusion = nn.ModuleList([Cross_attention(cfg.seq_encoder) for _ in range(cfg.seq_encoder.num_hidden_layers)])
        self.mlp = nn.Linear(cfg.encoder.node_feat_dim, cfg.encoder.node_feat_dim)
        self.esm_trans_layers = nn.Sequential(
            nn.Linear(1280, cfg.encoder.node_feat_dim*2), nn.ReLU(),
            nn.Linear(cfg.encoder.node_feat_dim*2, cfg.encoder.node_feat_dim), nn.ReLU(),
            nn.Linear(cfg.encoder.node_feat_dim, cfg.encoder.node_feat_dim)
        )
        
#         self.drop_out = nn.Dropout(0.1)

        self.transform = nn.Sequential(
            nn.Linear(cfg.encoder.node_feat_dim*2, cfg.encoder.node_feat_dim*2), nn.ReLU(),
            nn.Linear(cfg.encoder.node_feat_dim*2, cfg.encoder.node_feat_dim), nn.ReLU(),
            nn.Linear(cfg.encoder.node_feat_dim, cfg.encoder.node_feat_dim)
        )
        self.norm = LayerNorm(cfg.encoder.node_feat_dim)
        
        # prediction head
        if self.align:
            self.predictor = nn.Sequential(
                nn.Linear(cfg.encoder.node_feat_dim, cfg.encoder.node_feat_dim), nn.ReLU(),
                nn.Linear(cfg.encoder.node_feat_dim, cfg.encoder.node_feat_dim), nn.ReLU(),
                nn.Linear(cfg.encoder.node_feat_dim, cfg.num_classes)
            )

        self.num_classes = cfg.num_classes
        # Monotonic net
        class_dir = torch.tensor(cfg.class_dir)
        self.register_buffer('class_dir', class_dir)
        if self.num_classes>1:
            self.regular_layer = MonoRegularLayer(output_dim=self.num_classes)

    # v1
    def encode(self, batch):
        # residue特征
        receptor = batch['receptor']
        ligand = batch['ligand']
        
        ligand_embedding = self.aa_embedding(ligand['origin_aa'] if 'origin_aa' in ligand else ligand['aa'])
        receptor_embedding = self.aa_embedding(receptor['origin_aa'] if 'origin_aa' in receptor else receptor['aa'])
        
        x_lig = self.esm_trans_layers(ligand['esm_embedding']) + ligand_embedding * ligand['seq_mask'].unsqueeze(dim=-1)
        x_rec = self.esm_trans_layers(receptor['esm_embedding']) + receptor_embedding * receptor['seq_mask'].unsqueeze(dim=-1)
        
        # cross attention    
        for cross_att_layer in self.seq_fusion:
            x_lig = cross_att_layer(x_lig, ligand['seq_mask'], x_rec, receptor['seq_mask'])
            x_rec = cross_att_layer(x_rec, receptor['seq_mask'], x_lig, ligand['seq_mask'])

        return x_rec, x_lig
    
    def _get_reg_loss(self, preds, y):
        preds = self.regular_layer(preds * self.class_dir)
        losses = F.mse_loss(preds, (y * self.class_dir), reduction='none')
        return losses
    
    def _get_mse_loss(self, X, Y, mask=None):
        if mask is not None:
            criterion = nn.MSELoss(reduction='none')
            loss = criterion(X, Y)
            loss = (loss * mask.unsqueeze(dim=-1)).mean(dim=-1).sum() / mask.sum()
        else:
            criterion = nn.MSELoss(reduction='mean')
            loss = criterion(X, Y)
        return loss
    
    def _get_CosSim_loss(self, X, Y, mask=None):
        if mask is None:
            mask = 1
        loss = F.cosine_similarity(X,Y) * mask 
        loss = loss.sum()
        return loss    
    
    def forward(self, batch):
        x_rec, x_lig = self.encode(batch)    
        
        x_rec = (x_rec*batch['receptor']['seq_mask'][:,:,None]).max(dim=1)[0]
        x_lig = (x_lig*batch['ligand']['seq_mask'][:,:,None]).max(dim=1)[0]
        X = torch.concat([x_rec, x_lig], dim=-1)
        X = self.transform(X)
        X = self.norm(X)

        if self.align:
            self.struc_model.eval()
            seq_preds = self.struc_model.predictor(X).squeeze(-1)
        else:
            seq_preds = self.predictor(X).squeeze(-1)
        
        if self.align:
            with torch.no_grad():
                self.struc_model.eval()
                struc_feat, struc_pred = self.struc_model.get_model_feature(batch)
            sim_loss = - self._get_CosSim_loss(X, struc_feat) + self._get_mse_loss(X, struc_feat)
        else:
            sim_loss=torch.tensor(0)
            
        labels_mask = batch['receptor']['labels_mask']
        if seq_preds.size()==batch['receptor']['labels'].size():
            regression_loss = F.mse_loss(seq_preds, batch['receptor']['labels'], reduction='none')
            regression_loss = ((regression_loss * labels_mask).sum() / labels_mask.sum()) if labels_mask.sum()>0 else torch.tensor(0)
        else:
            regression_loss = F.mse_loss(seq_preds.unsqueeze(-1), batch['receptor']['labels'], reduction='none')
            regression_loss = (regression_loss * labels_mask).sum() / labels_mask.sum()

        if self.num_classes > 1:
            regular_loss = self._get_reg_loss(seq_preds, batch['receptor']['labels'])
            regular_loss = ((regular_loss[..., 1:] * labels_mask[..., 1:]).sum() / (labels_mask[..., 1:].sum().clip(1)))
        else:
            regular_loss = torch.tensor(0)
#         if torch.isnan(regression_loss) or regression_loss==0:
#             import pdb;pdb.set_trace()
        
        loss_dict = {
            'regression': regression_loss,
            'regular': regular_loss,
            'similar': sim_loss
        }
        if self.num_classes > 1:
            out_dict = {
                'dG_pred': seq_preds[:, 0],
                'dG_true': batch['receptor']['dG']
            }
        else:
            out_dict = {
                'dG_pred': seq_preds[:],
                'dG_true': batch['receptor']['dG']
            }
        
        return loss_dict, out_dict

    def infer(self, batch):
        x_rec, x_lig = self.encode(batch)
        x_rec = (x_rec * batch['receptor']['seq_mask'][:, :, None]).max(dim=1)[0]
        x_lig = (x_lig * batch['ligand']['seq_mask'][:, :, None]).max(dim=1)[0]
        X = torch.concat([x_rec, x_lig], dim=-1)
        X = self.transform(X)
        X = self.norm(X)

        self.struc_model.eval()
        seq_preds = self.struc_model.predictor(X).squeeze(-1)
        if self.num_classes > 1:
            return seq_preds[:, 0]
        else:
            return seq_preds

    def get_model_feature(self, batch):
        x_rec, x_lig = self.encode(batch)
        x_rec = (x_rec * batch['receptor']['seq_mask'][:, :, None]).max(dim=1)[0]
        x_lig = (x_lig * batch['ligand']['seq_mask'][:, :, None]).max(dim=1)[0]
        X = torch.concat([x_rec, x_lig], dim=-1)
        X = self.transform(X)
        X = self.norm(X)
        return X
