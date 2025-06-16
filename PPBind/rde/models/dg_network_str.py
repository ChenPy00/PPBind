# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


from rde.modules.encoders.single import PerResidueEncoder, PerAtomEncoder, PerResidueEncoder_aaindex1
from rde.modules.encoders.pair import ResiduePairEncoder, ResiduePairEncoder_rsaa, ResidueCrossPairEncoder_rsaa
from rde.modules.encoders.attn import GAEncoder, GAEncoderAtom, GCAEncoder
from rde.modules.mono.monotonic import MonoRegularLayer
from rde.utils.protein.constants import BBHeavyAtom
from .rde import CircularSplineRotamerDensityEstimator


# # 用非dG样本 不用kernel loss

class NetworkStr(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Pretrain
        try:
            ckpt = torch.load(cfg.rde_checkpoint, map_location='cpu')
        except:
            ckpt = torch.load('./trained_models/RDE.pt', map_location='cpu')
        self.rde = CircularSplineRotamerDensityEstimator(ckpt['config'].model)
        self.rde.load_state_dict(ckpt['model'])
        for p in self.rde.parameters():
            p.requires_grad_(False)
        res_dim = ckpt['config'].model.encoder.node_feat_dim
        self.convert_num = cfg.convert_num

        # Encoding
        # PerResidueEncoder 原版RDE
        # PerResidueEncoder_aaindex1为使用aaindex1的特征
        self.single_encoder = PerResidueEncoder_aaindex1(  
            feat_dim=cfg.encoder.node_feat_dim,
            # 原版
            # max_num_atoms=5,    # N, CA, C, O, CB,
            # LHQ改版：
            max_num_atoms=cfg.max_num_atoms,
        )

        self.single_fusion = nn.Sequential(
            nn.Linear(2 * res_dim, res_dim), nn.ReLU(),
            nn.Linear(res_dim, res_dim)
        )
        self.mut_bias = nn.Embedding(
            num_embeddings=2,
            embedding_dim=res_dim,
            padding_idx=0,
        )
        # Encoding
        # ResiduePairEncoder 原版RDE
        # ResiduePairEncoder_rsaa为使用rsaa的特征
        self.pair_encoder = ResiduePairEncoder_rsaa(
            feat_dim=cfg.encoder.pair_feat_dim,
            # 原版
            # max_num_atoms=5,    # N, CA, C, O, CB,
            # LHQ改版：
            max_num_atoms=cfg.max_num_atoms,
        )
        self.cross_pair_encoder = ResidueCrossPairEncoder_rsaa(
            feat_dim=cfg.encoder.pair_feat_dim,
            max_num_atoms=cfg.max_num_atoms,
        )
        self.attn_encoder = nn.ModuleList([GAEncoder(**cfg.encoder) for _ in range(self.convert_num)])
        self.crossattn_encoder = nn.ModuleList([GCAEncoder(**cfg.encoder) for _ in range(self.convert_num)])

        # Pred
        self.predictor = nn.Sequential(
            nn.Linear(res_dim, res_dim), nn.ReLU(),
            nn.Linear(res_dim, res_dim), nn.ReLU(),
            nn.Linear(res_dim, cfg.num_classes)
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

    def encode(self, batch_complex):
        # 分离受体和配体
        batch_entities = {}
        # 分别提取受体和配体的特征
        for name,batch in batch_complex.items():
            N, L = batch['aa'].shape[:2]
            mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
            chi = batch['chi'] * (1 - batch['mut_flag'].float())[:, :, None]

            x_single = self.single_encoder(
                aa=batch['aa'],
                phi=batch['phi'], phi_mask=batch['phi_mask'],
                psi=batch['psi'], psi_mask=batch['psi_mask'],
                chi=chi, chi_mask=batch['chi_mask'],
                mask_residue=mask_residue,
                rsaa=batch['RSAA'], 
                rsaa_mask=batch['RSAA_mask']
            )

            x_pret = self._encode_rde(batch)
            x = self.single_fusion(torch.cat([x_single, x_pret], dim=-1))
            b = self.mut_bias(batch['mut_flag'].long())  # 此处batch['mut_flag']是每个残基是否突变的mask
            x.add_(b)  # x = x + b

            z = self.pair_encoder(
                aa=batch['aa'],
                res_nb=batch['res_nb'], chain_nb=batch['chain_nb'],
                pos_atoms=batch['pos_atoms'], mask_atoms=batch['mask_atoms'],
                rsaa=batch['RSAA'], 
                rsaa_mask=batch['RSAA_mask']
            )
            
            batch_entities[name] = {'x': x, 'z': z, 'length':L,
                                    'pos_atoms':batch['pos_atoms'], 'mask_atoms':batch['mask_atoms'],'mask_residue':mask_residue}
        
        z_rec2lig = self.cross_pair_encoder(
            aa_q = batch_complex['receptor']['aa'],
            res_nb_q = batch_complex['receptor']['res_nb'], 
            chain_nb_q = batch_complex['receptor']['chain_nb'],
            pos_atoms_q = batch_complex['receptor']['pos_atoms'], 
            mask_atoms_q = batch_complex['receptor']['mask_atoms'],
            rsaa_q = batch_complex['receptor']['RSAA'], 
            rsaa_mask_q = batch_complex['receptor']['RSAA_mask'],
            #
            aa_kv = batch_complex['ligand']['aa'],
            res_nb_kv = batch_complex['ligand']['res_nb'], 
            chain_nb_kv = batch_complex['ligand']['chain_nb'],
            pos_atoms_kv = batch_complex['ligand']['pos_atoms'], 
            mask_atoms_kv = batch_complex['ligand']['mask_atoms'],
            rsaa_kv = batch_complex['ligand']['RSAA'], 
            rsaa_mask_kv = batch_complex['ligand']['RSAA_mask'],
        )
        
        z_lig2rec = self.cross_pair_encoder(
            aa_q = batch_complex['ligand']['aa'],
            res_nb_q = batch_complex['ligand']['res_nb'], 
            chain_nb_q = batch_complex['ligand']['chain_nb'],
            pos_atoms_q = batch_complex['ligand']['pos_atoms'], 
            mask_atoms_q = batch_complex['ligand']['mask_atoms'],
            rsaa_q = batch_complex['ligand']['RSAA'], 
            rsaa_mask_q = batch_complex['ligand']['RSAA_mask'],
            #
            aa_kv = batch_complex['receptor']['aa'],
            res_nb_kv = batch_complex['receptor']['res_nb'], 
            chain_nb_kv = batch_complex['receptor']['chain_nb'],
            pos_atoms_kv = batch_complex['receptor']['pos_atoms'], 
            mask_atoms_kv = batch_complex['receptor']['mask_atoms'],
            rsaa_kv = batch_complex['receptor']['RSAA'], 
            rsaa_mask_kv = batch_complex['receptor']['RSAA_mask'],
        )
        
        for i in range(self.convert_num):
            # 通过自注意力更新特征
            for name,batch in batch_entities.items():
                x = self.attn_encoder[i](
                    pos_atoms=batch['pos_atoms'],
                    res_feat=batch['x'], pair_feat=batch['z'],
                    mask=batch['mask_residue']
                )
                batch_entities[name]['x'] = x # batch['x'] = x
             
            # 通过交叉注意力更新特征
            batch_entities['receptor']['x'] = self.crossattn_encoder[i](
                    pos_atoms_q = batch_entities['receptor']['pos_atoms'],
                    res_feat_q = batch_entities['receptor']['x'],
                    mask_q = batch_entities['receptor']['mask_residue'], 
                    pos_atoms_kv = batch_entities['ligand']['pos_atoms'],
                    res_feat_kv = batch_entities['ligand']['x'],
                    mask_kv = batch_entities['ligand']['mask_residue'],
                    pair_feat = z_rec2lig,  
            )
            batch_entities['ligand']['x'] = self.crossattn_encoder[i](
                    pos_atoms_q = batch_entities['ligand']['pos_atoms'],
                    res_feat_q = batch_entities['ligand']['x'],
                    mask_q = batch_entities['ligand']['mask_residue'], 
                    pos_atoms_kv = batch_entities['receptor']['pos_atoms'],
                    res_feat_kv = batch_entities['receptor']['x'],
                    mask_kv = batch_entities['receptor']['mask_residue'],
                    pair_feat = z_lig2rec,  
            )
        return batch_entities['receptor'], batch_entities['ligand'] 

    def _get_reg_loss(self, preds, y):
        preds = self.regular_layer(preds * self.class_dir)
        losses = F.mse_loss(preds, (y * self.class_dir), reduction='none')
        return losses
    
    def _get_kernel_loss(self, y_pred, y_true, indices, sigma=0.8, epsilon=1e-9):
        diff = y_true-y_pred
        diff = (diff-diff[:, None])
        diff_ = diff[indices[0], indices[1]]
        
        def gaussian_kernel(diff, sigma):
            return (1/sigma)*torch.exp(-((diff/sigma)**2) / (2 * sigma ** 2)) + epsilon

        losses = -torch.log( gaussian_kernel(diff_, sigma) )
        return losses

    
    def get_loss_dict(self, preds, batch):   
        batch_size = batch['receptor']['aa'].size(0)
        labels_mask = batch['receptor']['labels_mask']

        loss_dict = {}
        
        # regression
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
        batch_rec, batch_lig = self.encode(batch)
        X = torch.concat([batch_rec['x']*batch_rec['mask_residue'][:,:,None], batch_lig['x']*batch_lig['mask_residue'][:,:,None]],dim=1).max(dim=1)[0]
        preds = self.predictor(X).squeeze(-1)
        
        loss_dict = self.get_loss_dict(preds, batch)

        out_dict = {
            'dG_pred': preds[:, 0] if self.num_classes > 1 else preds,
            'dG_true': batch['receptor']['dG']
        }
        
        return loss_dict, out_dict
    
    def get_model_feature(self, batch):
        
        batch_rec, batch_lig = self.encode(batch)
        X = torch.concat([batch_rec['x']*batch_rec['mask_residue'][:,:,None], batch_lig['x']*batch_lig['mask_residue'][:,:,None]],dim=1).max(dim=1)[0]
        preds = self.predictor(X).squeeze(-1)
        return X, preds

    def infer(self, batch):
        batch_rec, batch_lig = self.encode(batch)
        X = torch.concat([batch_rec['x']*batch_rec['mask_residue'][:,:,None], batch_lig['x']*batch_lig['mask_residue'][:,:,None]],dim=1).max(dim=1)[0]
        preds = self.predictor(X).squeeze(-1)
        return preds[:, 0] if self.num_classes > 1 else preds




