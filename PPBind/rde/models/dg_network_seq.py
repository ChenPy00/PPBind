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
from .rde import CircularSplineRotamerDensityEstimator
from rde.utils.protein.constants import BBHeavyAtom, get_aaindex1#AAindex1_matrix, AAindex1_matrix_dim
from rde.modules.encoders.seq import *

AAindex1_matrix = get_aaindex1(lowrank=False)
AAindex1_matrix_dim = AAindex1_matrix.size(1)


# +
class NetworkSeq(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # encoder模块
        self.aa_embedding = nn.Embedding(22, cfg.encoder.node_feat_dim)
        # 交叉注意力模块
        self.seq_fusion = nn.ModuleList([Cross_attention(cfg.seq_encoder) for _ in range(cfg.seq_encoder.num_hidden_layers)])
        self.mlp = nn.Linear(cfg.encoder.node_feat_dim, cfg.encoder.node_feat_dim)
        self.esm_trans_layers = nn.Sequential(
            nn.Linear(1280, cfg.encoder.node_feat_dim*2), nn.ReLU(),
            nn.Linear(cfg.encoder.node_feat_dim*2, cfg.encoder.node_feat_dim), nn.ReLU(),
            nn.Linear(cfg.encoder.node_feat_dim, cfg.encoder.node_feat_dim)
        )
        
#         self.drop_out = nn.Dropout(0.1)
    
        # prediction head模块
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

    def _encode_rde(self, batch, mask_extra=None):
        batch = {k: v for k, v in batch.items()}
        batch['chi_corrupt'] = batch['chi']
        batch['chi_masked_flag'] = batch['mut_flag']
        if mask_extra is not None:
            batch['mask_atoms'] = batch['mask_atoms'] * mask_extra[:, :, None]
        with torch.no_grad():
            return self.rde.encode(batch)
    
    # v1
    def encode(self, batch):
        # residue特征
        receptor = batch['receptor']
        ligand = batch['ligand']
        
        ligand_embedding = self.aa_embedding(ligand['aa'])
        receptor_embedding = self.aa_embedding(receptor['aa'])
        
        x_lig = self.esm_trans_layers(ligand['esm_embedding']) + ligand_embedding * ligand['mask'].unsqueeze(dim=-1)
        x_rec = self.esm_trans_layers(receptor['esm_embedding']) + receptor_embedding * receptor['mask'].unsqueeze(dim=-1)
        
        # cross attention    
        for cross_att_layer in self.seq_fusion:
            x_lig = cross_att_layer(x_lig, ligand['mask'], x_rec, receptor['mask'])
            x_rec = cross_att_layer(x_rec, receptor['mask'], x_lig, ligand['mask'])

        return x_rec, x_lig
    

    def _get_reg_loss(self, preds, y):
        preds = self.regular_layer(preds * self.class_dir)
        losses = F.mse_loss(preds, (y * self.class_dir), reduction='none')
        return losses
    
    
    def get_loss_dict(self, preds, batch):  
        batch_size = batch['receptor']['aa'].size(0)
        labels_mask = batch['receptor']['labels_mask']
        loss_dict = {}
        
        # regression
#         import pdb;pdb.set_trace()
        if preds.size()==batch['receptor']['labels'].size():
            # dG, log2er, Nkd
            regression_loss = F.mse_loss(preds, batch['receptor']['labels'], reduction='none')
        else:
            # dG
            regression_loss = F.mse_loss(preds.unsqueeze(-1), batch['receptor']['labels'], reduction='none')
        regression_loss = regression_loss * labels_mask
        loss_dict['regression'] = {'value': regression_loss, 'mask':labels_mask}
            
        # regular
        if self.num_classes > 1:
            # dG, log2er, Nkd
            regular_loss = self._get_reg_loss(preds, batch['receptor']['labels'])
            regular_loss = regular_loss[..., 1:] * labels_mask[..., 1:]
            loss_dict['regular'] = {'value': regular_loss, 'mask':labels_mask[..., 1:]}
        else:
            # dG
            regular_loss = torch.zeros((batch_size,)).to(regression_loss.device)
            regular_loss = torch.full((batch_size,), False).to(regression_loss.device)

        return loss_dict
    
    
    def forward(self, batch):
        x_rec, x_lig = self.encode(batch)
        X = x_lig.max(dim=1)[0]
        
        preds = self.predictor(X).squeeze(-1)
        
        loss_dict = self.get_loss_dict(preds, batch)

        out_dict = {
            'dG_pred': preds[:, 0] if self.num_classes > 1 else preds,
            'dG_true': batch['receptor']['dG']
        }
        
        return loss_dict, out_dict
    

