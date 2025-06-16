# -*- coding: utf-8 -*-
import torch
from Bio.PDB import Selection
from Bio.PDB.Residue import Residue
# from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PDBParser, DSSP, PDBIO
from Bio.PDB.MMCIFParser import MMCIFParser
from easydict import EasyDict

from .constants import (AA, max_num_heavyatoms,
                        restype_to_heavyatom_names, ressymb_to_resindex,
                        BBHeavyAtom)
from .icoord import get_chi_angles, get_backbone_torsions
resindex_to_ressymb = {v:k for k,v in ressymb_to_resindex.items()}


def patch_check_rsaa(dssp, struc_feat):
    # patch
    # dssp没记录某个/些残基，所以人工打补丁
    # 以达到长度相同
    # 通常长度上 dssp<=struc_feat

    # struc_feat_keys = set([(struc_feat.chain_id[i], (" ", struc_feat.resseq[i].item(), struc_feat.icode[i])) for i in range(len(struc_feat.aa))])
    struc_feat_dict = {
        (struc_feat.chain_id[i], (" ", struc_feat.resseq[i].item(), struc_feat.icode[i])):(
            struc_feat.resseq[i].item(), resindex_to_ressymb[struc_feat.aa[i].item()], '-',
            'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'
        ) for i in range(len(struc_feat.aa))
    }
    patch_key_list = set(struc_feat_dict.keys()) - set(dssp.keys())
    for patch_key in patch_key_list:
        dssp.property_dict[patch_key] = struc_feat_dict[patch_key]
            
    dssp.property_keys = list(dssp.property_dict.keys())
    dssp.property_list = list(dssp.property_dict.values())
    
    # 经过debug，发现dssp的顺序与struc_feat不同
    # 导致无法一一对比检查
    # 因此将check+get rsaa改写成change order+get rsaa
    # check
    
    # change order
    # 根据struc_feat中的顺序，读取dssp中的values，并获取rsaa
    try:
        RSAA = [dssp.property_dict[k][3] for k in struc_feat_dict.keys()]
        RSAA_value = [rsaa if type(rsaa)==float else 0 for rsaa in RSAA]
        RSAA_mask = [True if type(rsaa)==float else False for rsaa in RSAA ]
        return (RSAA_value, RSAA_mask)
    except:
        return False


def find_problem_res(model):
    '''
    这里说的问题残基，是指同一个位点既有非标准残基又有标准残基
    '''
    problem_dict = {}
    for chain in model.get_chains():
        chain_dict = {}
        for res in chain.get_residues():
            if res.id[1] not in chain_dict:
                chain_dict[res.id[1]]=[]
                chain_dict[res.id[1]].append(res)
            else:
                chain_dict[res.id[1]].append(res)
        problem_pos = {k:v for k,v in chain_dict.items() if len(v)>1}
        problem_dict[chain.id] = problem_pos
    return problem_dict


def clean_pdb(model, problem_dict):
    '''
    既有非标准残基又有标准残基的位点， 只保留标准残基
    '''
    clean_model = model.copy()
    for chain, problem_pos in problem_dict.items():
        if len(problem_pos)>0:
            for pos, res_list in problem_pos.items():
                het_res_list = [r for r in problem_dict[chain][pos] if r.id[0]!=' ']
                for het_res in het_res_list:
                    try:
                        clean_model.child_dict[chain].detach_child(het_res.id)
                    except:
                        import pdb;pdb.set_trace()
    return clean_model


def _get_residue_heavyatom_info(res: Residue):
    pos_heavyatom = torch.zeros([max_num_heavyatoms, 3], dtype=torch.float)
    mask_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.bool)
    bfactor_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.float)
    restype = AA(res.get_resname())
    for idx, atom_name in enumerate(restype_to_heavyatom_names[restype]):
        if atom_name == '': continue
        if atom_name in res:
            pos_heavyatom[idx] = torch.tensor(res[atom_name].get_coord().tolist(), dtype=pos_heavyatom.dtype)
            mask_heavyatom[idx] = True
            bfactor_heavyatom[idx] = res[atom_name].get_bfactor()
    return pos_heavyatom, mask_heavyatom, bfactor_heavyatom


def parse_pdb(path, model_id, unknown_threshold=1.0):
    parser = PDBParser()
    structure = parser.get_structure(None, path)
    return parse_biopython_structure(structure[model_id], unknown_threshold=unknown_threshold)


def parse_mmcif_assembly(path, model_id, assembly_id=0, unknown_threshold=1.0):
    parser = MMCIFParser()
    structure = parser.get_structure(None, path)
    mmcif_dict = parser._mmcif_dict
    if '_pdbx_struct_assembly_gen.asym_id_list' not in mmcif_dict:
        return parse_biopython_structure(structure[model_id], unknown_threshold=unknown_threshold)
    else:
        assemblies = [tuple(chains.split(',')) for chains in mmcif_dict['_pdbx_struct_assembly_gen.asym_id_list']]
        label_to_auth = {}
        for label_asym_id, auth_asym_id in zip(mmcif_dict['_atom_site.label_asym_id'], mmcif_dict['_atom_site.auth_asym_id']):
            label_to_auth[label_asym_id] = auth_asym_id
        model_real = list({structure[model_id][label_to_auth[ch]] for ch in assemblies[assembly_id]})
        return parse_biopython_structure(model_real)


def parse_biopython_structure(model, pdb_path, unknown_threshold=1.0):
    '''
    获取一个model(PDBParser get_structure读取一个pdb文件后得到)中RDE所需信息。
    '''
    problem_dict = find_problem_res(model)
    if len(problem_dict)>=1:
        clean_model = clean_pdb(model, problem_dict)
        io = PDBIO()
        io.set_structure(clean_model)
        io.save("./tmp_model.pdb")
        dssp = DSSP(clean_model, "./tmp_model.pdb")
    else:
        clean_model = model
        dssp = DSSP(model, pdb_path)
    
    chains = Selection.unfold_entities(clean_model, 'C')
    chains.sort(key=lambda c: c.get_id())
    data = EasyDict({
        'chain_id': [], 'chain_nb': [],
        'resseq': [], 'icode': [], 'res_nb': [],
        'aa': [],
        'pos_heavyatom': [], 'mask_heavyatom': [],
        'bfactor_heavyatom': [],
        'phi': [], 'phi_mask': [],
        'psi': [], 'psi_mask': [],
        'chi': [], 'chi_alt': [], 'chi_mask': [], 'chi_complete': [],
    })
    tensor_types = {
        'chain_nb': torch.LongTensor,
        'resseq': torch.LongTensor,
        'res_nb': torch.LongTensor,
        'aa': torch.LongTensor,
        'pos_heavyatom': torch.stack,
        'mask_heavyatom': torch.stack,
        'bfactor_heavyatom': torch.stack,

        'phi': torch.FloatTensor,
        'phi_mask': torch.BoolTensor,
        'psi': torch.FloatTensor,
        'psi_mask': torch.BoolTensor,

        'chi': torch.stack,
        'chi_alt': torch.stack,
        'chi_mask': torch.stack,
        'chi_complete': torch.BoolTensor,
    }

    count_aa, count_unk = 0, 0

    for i, chain in enumerate(chains):
        chain.atom_to_internal_coordinates()
        seq_this = 0   # Renumbering residues
        residues = Selection.unfold_entities(chain, 'R')
        residues.sort(key=lambda res: (res.get_id()[1], res.get_id()[2]))   # Sort residues by resseq-icode
        for _, res in enumerate(residues):
            resname = res.get_resname()
            if not AA.is_aa(resname): continue
            if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')): continue
            restype = AA(resname)
            count_aa += 1
            if restype == AA.UNK: 
                count_unk += 1
                continue

            # Chain info
            data.chain_id.append(chain.get_id())
            data.chain_nb.append(i)

            # Residue types
            data.aa.append(restype) # Will be automatically cast to torch.long

            # Heavy atoms
            pos_heavyatom, mask_heavyatom, bfactor_heavyatom = _get_residue_heavyatom_info(res)
            data.pos_heavyatom.append(pos_heavyatom)
            data.mask_heavyatom.append(mask_heavyatom)
            data.bfactor_heavyatom.append(bfactor_heavyatom)

            # Backbone torsions
            phi, psi, _ = get_backbone_torsions(res)
            if phi is None:
                data.phi.append(0.0)
                data.phi_mask.append(False)
            else:
                data.phi.append(phi)
                data.phi_mask.append(True)
            if psi is None:
                data.psi.append(0.0)
                data.psi_mask.append(False)
            else:
                data.psi.append(psi)
                data.psi_mask.append(True)

            # Chi
            chi, chi_alt, chi_mask, chi_complete = get_chi_angles(restype, res)
            data.chi.append(chi)
            data.chi_alt.append(chi_alt)
            data.chi_mask.append(chi_mask)
            data.chi_complete.append(chi_complete)

            # Sequential number
            resseq_this = int(res.get_id()[1])
            icode_this = res.get_id()[2]
            if seq_this == 0:
                seq_this = 1
            else:
                d_CA_CA = torch.linalg.norm(data.pos_heavyatom[-2][BBHeavyAtom.CA] - data.pos_heavyatom[-1][BBHeavyAtom.CA], ord=2).item()
                if d_CA_CA <= 4.0:
                    seq_this += 1
                else:
                    d_resseq = resseq_this - data.resseq[-1]
                    seq_this += max(2, d_resseq)

            data.resseq.append(resseq_this)
            data.icode.append(icode_this)
            data.res_nb.append(seq_this)

    if len(data.aa) == 0:
        return None, None

    if (count_unk / count_aa) >= unknown_threshold:
        return None, None

    seq_map = {}
    # data.chain_id: 每个氨基酸所在链的ID
    # data.resseq: 每个氨基酸在链中的序号
    # data.icode: 每个氨基酸的icode，也可以是空格
    for i, (chain_id, resseq, icode) in enumerate(zip(data.chain_id, data.resseq, data.icode)):
        seq_map[(chain_id, resseq, icode)] = i

    for key, convert_fn in tensor_types.items():
        data[key] = convert_fn(data[key])
    
    # 记录RSAA
    result = patch_check_rsaa(dssp, data)
    if not result:
        print(f'PDB:{pdb_path} is wrong  RASS')
    else:
        RSAA_value, RSAA_mask = result
        data['RSAA'] = torch.FloatTensor(RSAA_value)
        data['RSAA_mask'] = torch.BoolTensor(RSAA_mask)
    if len(data['RSAA'])!=len(data['aa']):
        import pdb
        pdb.set_trace()
    
    return data, seq_map
