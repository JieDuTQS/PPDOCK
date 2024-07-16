import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Dataset
import lmdb
import pickle
from torch_geometric.data import HeteroData
import torch
import numpy as np
import pandas as pd
import scipy.spatial
from torch_geometric.data import HeteroData

def compute_dis_between_two_vector(a, b):
    return (((a - b)**2).sum())**0.5

def get_keepNode(compound_coords_mean=None, protein_node_xyz=None, protein_node_num=None, pocket_radius=None, add_noise_to_poc_com=None):
    pocketNode = np.zeros(protein_node_num, dtype=bool)
    if add_noise_to_poc_com==0:
        com=compound_coords_mean
    elif add_noise_to_poc_com==1:
        com = compound_coords_mean + 5.0 * (2 * np.random.rand(*compound_coords_mean.shape) - 1)
    for i, node in enumerate(protein_node_xyz):
        dis = compute_dis_between_two_vector(node, com)
        pocketNode[i] = dis < pocket_radius 
    return pocketNode

def My_get_keepNode(args=None,compound_coords_center=None, compound_coords=None,protein_node_xyz=None,add_noise=None):
    compound_coords_center=torch.tensor(compound_coords_center).float().unsqueeze(dim=0)
    compound_coords=torch.tensor(compound_coords).float()
    protein_node_xyz=protein_node_xyz.float()
    protein_node_num=protein_node_xyz.shape[0]
    pocket_percent_num=int(protein_node_num*args.pocket_getting_percent)
    pocket_num_max=max(pocket_percent_num,args.pocket_minimum_quantity)
    pocketNode = torch.zeros(protein_node_num, dtype=bool)
    pocket_radius=args.pocket_radius_minimum
    if args.pocket_obtaining_method=='compound_coords_center':
        com=compound_coords_center
        if add_noise:
            com = compound_coords_center + args.add_noise_center * (2 * np.random.rand(*compound_coords_center.shape) - 1)
            com=com.float()
        while torch.sum(pocketNode)< pocket_num_max:
            distances_map=torch.cdist(protein_node_xyz, com)
            distances_map_bool=distances_map<pocket_radius
            pocketNode=distances_map_bool.squeeze()
            pocket_radius+=args.radius_distance_interval
    elif args.pocket_obtaining_method=='dis_map_pro_com':
        while torch.sum(pocketNode)< pocket_num_max:
            distances_map=torch.cdist(protein_node_xyz, compound_coords)
            distances_map_bool=distances_map<pocket_radius
            distances_map_bool=torch.sum(distances_map_bool,dim=1)
            pocketNode=distances_map_bool>0
            pocket_radius+=args.radius_distance_interval  
    pocketNode_out=pocketNode.numpy()
    return pocketNode_out

def uniform_random_rotation(x):
    def generate_random_z_axis_rotation():
        R = np.eye(3)
        x1 = np.random.rand()
        R[0, 0] = R[1, 1] = np.cos(2 * np.pi * x1)
        R[0, 1] = -np.sin(2 * np.pi * x1)
        R[1, 0] = np.sin(2 * np.pi * x1)
        return R
    x2 = 2 * np.pi * np.random.rand()
    x3 = np.random.rand()
    R = generate_random_z_axis_rotation()
    v = np.array([
        np.cos(x2) * np.sqrt(x3),
        np.sin(x2) * np.sqrt(x3),
        np.sqrt(1 - x3)
    ])
    H = np.eye(3) - (2 * np.outer(v, v))
    M = -(H @ R)
    x = x.reshape((-1, 3))
    mean_coord = np.mean(x, axis=0)
    return ((x - mean_coord) @ M) + mean_coord @ M

def construct_data_graph(args=None, 
                            protein_node_xyz=None, protein_seq=None, 
                            protein_esm2_feat=None,
                            rdkit_coords=None,
                            compound_coords=None, compound_node_features=None, compound_edge_list=None, compound_edge_attr_list=None,compound_LAS_edge_index=None,
                            add_noise_to_poc_com=None,  
                            random_rotation=None,
                            pdb_id=None, group=None, seed=None,compound_center_pred=None):
    protein_node_num = protein_node_xyz.shape[0] 
    protein_node_xyz_mean = protein_node_xyz.mean(dim=0) 
    compound_coords = compound_coords - protein_node_xyz_mean.numpy() 
    protein_node_xyz = protein_node_xyz - protein_node_xyz_mean 
    compound_coords_mean = compound_coords.mean(axis=0) 
    if compound_center_pred is not None:
        compound_coords_mean_pred_for_pocket = compound_center_pred
    else:
        compound_coords_mean_pred_for_pocket=compound_coords_mean
    if args.pocket_mode=='FABind_pocket_mode':
        if group == 'train': 
            pocketNode = get_keepNode(compound_coords_mean_pred_for_pocket, protein_node_xyz.numpy(), protein_node_num, args.pocket_radius, add_noise_to_poc_com)
        else:
            pocketNode=get_keepNode(compound_coords_mean_pred_for_pocket, protein_node_xyz.numpy(), protein_node_num, args.pocket_radius, 0)
        pocketNode_no_noise = get_keepNode(compound_coords_mean_pred_for_pocket, protein_node_xyz.numpy(), protein_node_num, args.pocket_radius, 0)    
    elif args.pocket_mode=='My_pocket_mode':
        if group == 'train': 
            pocketNode = My_get_keepNode(args,compound_coords_mean_pred_for_pocket, compound_coords,protein_node_xyz,args.add_noise)
        else:
            pocketNode=My_get_keepNode(args,compound_coords_mean_pred_for_pocket, compound_coords,protein_node_xyz,False)
        pocketNode_no_noise = My_get_keepNode(args,compound_coords_mean_pred_for_pocket, compound_coords,protein_node_xyz,False)
    if pocketNode.sum() < 5:
        pocketNode[:100] = True
    pocket_node_xyz = protein_node_xyz[pocketNode] 
    data = HeteroData()
    dis_map_poc_com = scipy.spatial.distance.cdist(pocket_node_xyz.cpu().numpy(), compound_coords)
    dis_map_poc_com[dis_map_poc_com>args.interactionThresholdDistance] = args.interactionThresholdDistance
    data.dis_map_poc_com = torch.tensor(dis_map_poc_com, dtype=torch.float).flatten()
    data.pocket_node_xyz = pocket_node_xyz 
    data.compound_coords = torch.tensor(compound_coords, dtype=torch.float) 
    if torch.is_tensor(protein_esm2_feat):
        data['pocket'].node_feats = protein_esm2_feat[pocketNode] 
    else:
        raise ValueError("protein_esm2_feat should be a tensor")
    data['pocket'].pocketNode = torch.tensor(pocketNode, dtype=torch.bool) 
    data['compound'].node_feats = compound_node_features.float() 
    data['compound', 'LAS', 'compound'].edge_index = compound_LAS_edge_index 
    num_pocket = pocket_node_xyz.shape[0] 
    num_compound = compound_node_features.shape[0]
    if args.rdkit_coords_init_mode == 'pocket_center_rdkit':
        if random_rotation: 
            rdkit_coords = torch.tensor(uniform_random_rotation(rdkit_coords)) 
        else:
            rdkit_coords = torch.tensor(rdkit_coords)
        rdkit_coords_init = rdkit_coords - rdkit_coords.mean(dim=0).reshape(1, 3) + pocket_node_xyz.mean(dim=0).reshape(1, 3)
    data['complex'].node_coords = torch.cat( 
        (
            torch.zeros(1, 3),
            rdkit_coords_init, 
            torch.zeros(1, 3), 
            pocket_node_xyz 
        ), dim=0
    ).float()
    data['complex'].node_coords_LAS = torch.cat( 
        (
            torch.zeros(1, 3),
            rdkit_coords,
            torch.zeros(1, 3), 
            torch.zeros_like(pocket_node_xyz)
        ), dim=0
    ).float()
    segment = torch.zeros(num_pocket + num_compound + 2) 
    segment[num_compound+1:] = 1 
    data['complex'].segment = segment 
    mask = torch.zeros(num_pocket + num_compound + 2)
    mask[:num_compound+2] = 1 
    data['complex'].mask = mask.bool() 
    is_global = torch.zeros(num_pocket + num_compound + 2)
    is_global[0] = 1
    is_global[num_compound+1] = 1
    data['complex'].is_global = is_global.bool()
    data['complex', 'c2c', 'complex'].edge_index = compound_edge_list[:,:2].long().t().contiguous() + 1
    data['complex', 'LAS', 'complex'].edge_index = compound_LAS_edge_index + 1
    data['compound'].rdkit_coords_init = rdkit_coords_init 
    data['compound'].rdkit_coords = rdkit_coords 
    data['compound_edge_list'].x = (compound_edge_list[:,:2].long().contiguous() + 1).clone() 
    data['compound_LAS_edge_list'].x = data['complex', 'LAS', 'complex'].edge_index.clone().t()  
    data.protein_node_xyz = protein_node_xyz
    data.compound_coords_mean = torch.tensor(compound_coords_mean, dtype=torch.float).unsqueeze(0) 
    data.protein_seq = protein_seq
    data.protein_node_xyz_mean = protein_node_xyz_mean.unsqueeze(0) 
    data.pocketNode_no_noise = torch.tensor(pocketNode_no_noise, dtype=torch.int) 
    if torch.is_tensor(protein_esm2_feat):
        data['protein'].node_feats = protein_esm2_feat 
    else:
        raise ValueError("protein_esm2_feat should be a tensor")      
    return data

class MYDataSet(Dataset):
    def __init__(self,root, transform=None, pre_transform=None, pre_filter=None,args=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.args=args
        print('self.processed_paths:',self.processed_paths)
        self.compound_dict = lmdb.open(self.processed_paths[0], readonly=True, max_readers=1, lock=False, readahead=False, meminit=False) 
        self.protein_esm2_feat = lmdb.open(self.processed_paths[1], readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        self.protein_dict = lmdb.open(self.processed_paths[2], readonly=True, max_readers=1, lock=False, readahead=False, meminit=False) 
        self.compound_rdkit_coords = torch.load(self.processed_paths[3]) 
        self.data = torch.load(self.processed_paths[4])
        self.compound_center = torch.load(self.processed_paths[5]) 
        self.add_noise_to_poc_com = self.args.add_noise_to_poc_com
        self.seed = self.args.seed
    @property
    def processed_file_names(self):
        return ['compound_LAS_edge_index.lmdb', 'esm2_t33_650M_UR50D.lmdb','protein_1d_3d.lmdb','compound_rdkit_coords.pt','data.pt','anchor.pt']
    def len(self):
        return len(self.data)
    def get(self, idx):
        line = self.data.iloc[idx]
        group = line['group'] if "group" in line.index else 'train'
        if group == 'train': 
            add_noise_to_poc_com = self.add_noise_to_poc_com 
            random_rotation = True
        else:
            add_noise_to_poc_com = 0
            random_rotation = False
        protein_name = line['protein_name'] 
        with self.protein_dict.begin() as txn: 
            protein_node_xyz, protein_seq= pickle.loads(txn.get(protein_name.encode()))
        with self.protein_esm2_feat.begin() as txn: 
            protein_esm2_feat = pickle.loads(txn.get(protein_name.encode()))
        compound_name = line['compound_name'] 
        rdkit_coords = self.compound_rdkit_coords[compound_name] 
        try:
            compound_center_pred = self.compound_center[compound_name][0]
        except:
            compound_center_pred=None
        with self.compound_dict.begin() as txn:
            compound_coords, compound_node_features, compound_edge_list, compound_edge_attr_list, pair_dis_distribution, compound_LAS_edge_index = pickle.loads(txn.get(compound_name.encode()))
        data = construct_data_graph(self.args, 
                                    protein_node_xyz, protein_seq, 
                                    protein_esm2_feat,
                                    rdkit_coords,
                                    compound_coords, compound_node_features, compound_edge_list, compound_edge_attr_list,compound_LAS_edge_index,
                                    add_noise_to_poc_com,  
                                    random_rotation,
                                    pdb_id=compound_name, group=group, seed=self.seed,compound_center_pred=compound_center_pred)
        data.pdb = line['pdb'] if "pdb" in line.index else f'smiles_{idx}' 
        data.group = group 
        return data

def get_data(args, logger):
    logger.log_message(f"Loading dataset")
    logger.log_message(f"compound feature based on torchdrug")
    logger.log_message(f"protein feature based on esm2")
    new_dataset = MYDataSet(root=args.data_path,args=args)
    train_tmp = new_dataset.data.query("c_length < 100 and native_num_contact > 5 and group =='train' and use_compound_com").reset_index(drop=True)
    valid_test_tmp = new_dataset.data.query("(group == 'valid' or group == 'test') and use_compound_com").reset_index(drop=True)
    new_dataset.data = pd.concat([train_tmp, valid_test_tmp], axis=0).reset_index(drop=True)
    d = new_dataset.data
    only_native_train_index = d.query("group =='train'").index.values
    train = new_dataset[only_native_train_index]
    valid_index = d.query("group =='valid'").index.values
    valid = new_dataset[valid_index]
    test_index = d.query("group =='test'").index.values
    test = new_dataset[test_index]
    return train, valid, test