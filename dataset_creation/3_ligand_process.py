# 'compound_LAS_edge_index.lmdb','compound_rdkit_coords.pt'
import torch
import argparse
import os
import pandas as pd
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem import AllChem
import scipy.spatial
from torch_geometric.utils import dense_to_sparse
from rdkit.Geometry import Point3D
import numpy as np
import torch
from torchdrug import data as td
import lmdb
from tqdm import tqdm
import pickle
import copy


def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

#adj - > n_hops connections adj
def n_hops_adj(adj, n_hops):
    adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

    for i in range(2, n_hops+1):
        adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
    extend_mat = torch.zeros_like(adj)

    for i in range(1, n_hops+1):
        extend_mat += (adj_mats[i] - adj_mats[i-1]) * i

    return extend_mat

# mol_mask[i][j] = 1 means that atom i and atom j are  
# connected by a bond(origin adjacent matrix), or 2-hop away, or in the same ring structure
def get_LAS_distance_constraint_mask(mol):
    # Get the adj
    adj = Chem.GetAdjacencyMatrix(mol)
    adj = torch.from_numpy(adj)
    extend_adj = n_hops_adj(adj,2)
    # add ring
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        # print(ring)
        for i in ring:
            for j in ring:
                if i==j:
                    continue
                else:
                    extend_adj[i][j]+=1
    # turn to mask
    mol_mask = binarize(extend_adj)
    return mol_mask

def get_compound_pair_dis_distribution(coords, LAS_distance_constraint_mask=None):
    pair_dis = scipy.spatial.distance.cdist(coords, coords)
    bin_size=1
    bin_min=-0.5
    bin_max=15
    if LAS_distance_constraint_mask is not None:
        if pair_dis is None:
            print(coords)
            print(coords.shape)

        pair_dis[LAS_distance_constraint_mask==0] = bin_max
        # diagonal is zero.
        for i in range(pair_dis.shape[0]):
            pair_dis[i, i] = 0
    pair_dis = torch.tensor(pair_dis, dtype=torch.float)
    pair_dis[pair_dis>bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=16)
    pair_dis_distribution = pair_dis_one_hot.float()
    return pair_dis_distribution

def extract_torchdrug_feature_from_mol(mol, has_LAS_mask=False, ):
    # N x 3
    coords = mol.GetConformer().GetPositions()
    if has_LAS_mask:
        # N x N
        LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol)
        LAS_edge_index, _ = dense_to_sparse(LAS_distance_constraint_mask)
    else:
        LAS_distance_constraint_mask = None
        LAS_edge_index = None
    
    molstd = td.Molecule.from_smiles(Chem.MolToSmiles(mol),node_feature='property_prediction')
    compound_node_features = molstd.node_feature # nodes_chemical_features
    edge_list = molstd.edge_list # [num_edge, 3] (node_in, node_out, relation)
    edge_weight = molstd.edge_weight # [num_edge, 1]
    assert edge_weight.max() == 1
    assert edge_weight.min() == 1
    assert coords.shape[0] == compound_node_features.shape[0]
    x = [coords, compound_node_features, edge_list, LAS_edge_index]
    return x

def read_mol_and_renumber(sdf_fileName, mol2_fileName, verbose=False):
    # read mol
    problem = False
    mol = Chem.MolFromMolFile(sdf_fileName, sanitize=False)
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        sm = Chem.MolToSmiles(mol)
    except Exception as e:
        problem = True
    if problem:
        mol = Chem.MolFromMol2File(mol2_fileName, sanitize=False)
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
            sm = Chem.MolToSmiles(mol)
            problem = False
        except Exception as e:
            sm = str(e)
            problem = True
    
    if problem:
        return None
    
    # renumber atoms
    m_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
    mol = Chem.RenumberAtoms(mol, m_order)
    
    return mol

def read_smiles(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
    except:
        print("warning: cannot sanitize smiles: ", smile)
        mol = Chem.MolFromSmiles(smile, sanitize=False)
    
    sm = Chem.MolToSmiles(mol)
    m_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
    mol = Chem.RenumberAtoms(mol, m_order)
    Chem.SanitizeMol(mol)
    return mol

def generate_conformation(mol):
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv2()
    try:
        rid = AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    except:
        mol.Compute2DCoords()
    mol = Chem.RemoveHs(mol)
    return mol

def save_to_lmdb(lmdb_path, data_name,ligand_data):
    env = lmdb.open(lmdb_path, map_size=1e12) 
    

    with env.begin(write=True) as txn:
        txn.put(data_name.encode('utf-8'), pickle.dumps(ligand_data))
    env.close()  
    
        

def process_data(txt_file_path, data_folder_path, output_folder_path, output_file_name):
    with open(txt_file_path, 'r') as f:
        data_names = [line.strip() for line in f]
    
    result_dict = {}
    lmdb_path = os.path.join(output_folder_path, 'compound_LAS_edge_index.lmdb')
    for data_name in tqdm(data_names):
        subfolder_path = os.path.join(data_folder_path, data_name)
        if not os.path.isdir(subfolder_path):
            print(f"No subfolder: {subfolder_path}")
            print()
            continue
        
        sdf_file_path = None
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('_ligand_reorder.sdf'):
                sdf_file_path = os.path.join(subfolder_path, file_name)
                break
        
        mol = Chem.MolFromMolFile(sdf_file_path)
        mol=Chem.RemoveHs(mol)
        real_coords=mol.GetConformer().GetPositions()
        
        mol_rdkit = copy.deepcopy(mol)
        mol_rdkit.RemoveAllConformers()
        mol_rdkit = generate_conformation(mol_rdkit) 
        molecule_info = extract_torchdrug_feature_from_mol(mol_rdkit, has_LAS_mask=True)            

        compound_coords=real_coords
        compound_node_features=torch.tensor(molecule_info[1])
        compound_edge_list=torch.tensor(molecule_info[2])
        
        compound_edge_attr_list=torch.rand((compound_edge_list.size()[0],19))
        pair_dis_distribution=torch.rand((compound_node_features.size()[0],compound_node_features.size()[0],16))
        
        compound_LAS_edge_index=torch.tensor(molecule_info[3])
        
        ligand_data = (compound_coords, compound_node_features,compound_edge_list,compound_edge_attr_list,pair_dis_distribution,compound_LAS_edge_index)
        save_to_lmdb(lmdb_path, data_name,ligand_data)

        rdkit_coords=molecule_info[0]
        result_dict[data_name] = rdkit_coords
        if len(compound_coords)!=len(rdkit_coords):
            print('\n\n\n')
            print('compound_coords:',len(compound_coords))
            print('rdkit_coords:',len(rdkit_coords),'\n\n\n')
    output_file_path = os.path.join(output_folder_path, output_file_name)
    torch.save(result_dict, output_file_path)
    print(f"saved to: {output_file_path}")
    

txt_file_path = 'data_ori/list.txt'
data_folder_path = 'data_ori/test_data'
output_folder_path = 'data_processed'
output_file_name = 'compound_rdkit_coords.pt'

process_data(txt_file_path, data_folder_path, output_folder_path, output_file_name)



