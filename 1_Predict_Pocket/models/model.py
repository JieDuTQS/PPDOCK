import torch
from torch_geometric.utils import to_dense_batch
from torch import nn
from models.Anchor_Prediction_Model import EfficientMCAttModel
from torch_scatter import scatter_mean
from models.pointnet import PointNetEncoder


class model_FABind_layer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args=args
        self.embedding_channels=args.location_hidden_size
        self.layernorm = torch.nn.LayerNorm(self.embedding_channels) 
        
        self.coordinate_scale = args.coordinate_scale 
        self.normalize_coord = lambda x: x / self.coordinate_scale 
        self.unnormalize_coord = lambda x: x * self.coordinate_scale 

        n_channel = 1 
        self.location_pred_model = EfficientMCAttModel(
            args, self.embedding_channels, self.embedding_channels, n_channel, n_edge_feats=0, n_layers=args.location_num_layers, n_iter=args.location_num_iters,
            inter_cutoff=args.inter_cutoff, intra_cutoff=args.intra_cutoff, normalize_coord=self.normalize_coord, unnormalize_coord=self.unnormalize_coord,
        )

        self.glb_c = nn.Parameter(torch.ones(1, self.embedding_channels))
        self.glb_p = nn.Parameter(torch.ones(1, self.embedding_channels))
        protein_feat_size = args.protein_feat_size
        compound_feat_size = args.compound_feat_size        
        
        if args.location_pred_type=='EGNN_mean':
            self.location_protein_input_linear = nn.Linear(protein_feat_size,  self.embedding_channels) 
            self.location_compound_input_linear = nn.Linear(compound_feat_size,  self.embedding_channels) 
                    
            torch.nn.init.xavier_uniform_(self.location_protein_input_linear.weight, gain=0.001)
            torch.nn.init.xavier_uniform_(self.location_compound_input_linear.weight, gain=0.001)    
                    
        elif args.location_pred_type=='pointnet_EGNN':
            self.location_protein_input_linear = nn.Linear(protein_feat_size,  self.embedding_channels) 
            self.location_compound_input_linear = nn.Linear(args.compound_global_feat_size,  self.embedding_channels) 
            
            self.compound_global_feat_pointnet=PointNetEncoder(channel=args.compound_feat_size)

            torch.nn.init.xavier_uniform_(self.location_protein_input_linear.weight, gain=0.001)
            torch.nn.init.xavier_uniform_(self.location_compound_input_linear.weight, gain=0.001)
            
        elif args.location_pred_type=='EGNN_pointnet':
            self.location_protein_input_linear = nn.Linear(protein_feat_size,  self.embedding_channels) 
            self.location_compound_input_linear = nn.Linear(compound_feat_size,  self.embedding_channels) 
            self.compound_global_crood_pointnet=PointNetEncoder(channel=3)
            self.compound_coord_center_linear = nn.Linear(1024, 3)
                                
            torch.nn.init.xavier_uniform_(self.location_protein_input_linear.weight, gain=0.001)
            torch.nn.init.xavier_uniform_(self.location_compound_input_linear.weight, gain=0.001)  
                    
    def forward(self, data):
        if self.args.location_pred_type=='EGNN_mean':
            compound_batch = data['compound'].batch
            protein_batch = data['protein'].batch
            complex_batch_whole_protein = data['complex_whole_protein'].batch

            batched_complex_coord_whole_protein = self.normalize_coord(data['complex_whole_protein'].node_coords.unsqueeze(-2)) 
            batched_complex_coord_LAS_whole_protein = self.normalize_coord(data['complex_whole_protein'].node_coords_LAS.unsqueeze(-2)) 
            batched_compound_emb_whole_protein = self.location_compound_input_linear(data['compound'].node_feats) 
            batched_protein_emb_whole_protein = self.location_protein_input_linear(data['protein'].node_feats) 


            for i in range(complex_batch_whole_protein.max()+1):
                if i == 0:
                    new_samples_whole_protein = torch.cat((
                        self.glb_c, batched_compound_emb_whole_protein[compound_batch==i], 
                        self.glb_p, batched_protein_emb_whole_protein[protein_batch==i]
                        ), dim=0)
                else:
                    new_sample_whole_protein = torch.cat((
                        self.glb_c, batched_compound_emb_whole_protein[compound_batch==i], 
                        self.glb_p, batched_protein_emb_whole_protein[protein_batch==i]
                        ), dim=0)
                    new_samples_whole_protein = torch.cat((new_samples_whole_protein, new_sample_whole_protein), dim=0)

            complex_coords_whole_protein, complex_out_whole_protein = self.location_pred_model(
                batched_complex_coord_whole_protein,
                new_samples_whole_protein, 
                batch_id=complex_batch_whole_protein, 
                segment_id=data['complex_whole_protein'].segment,
                mask=data['complex_whole_protein'].mask, 
                is_global=data['complex_whole_protein'].is_global,
                compound_edge_index=data['complex_whole_protein', 'c2c', 'complex_whole_protein'].edge_index,
                LAS_edge_index=data['complex_whole_protein', 'LAS', 'complex_whole_protein'].edge_index,
                batched_complex_coord_LAS=batched_complex_coord_LAS_whole_protein,
                LAS_mask=None
            )
            
            complex_coords_whole_protein=self.unnormalize_coord(complex_coords_whole_protein)
            compound_flag_whole_protein = torch.logical_and(data['complex_whole_protein'].segment == 0, ~data['complex_whole_protein'].is_global) 
            compound_coords_out_whole_protein = complex_coords_whole_protein[compound_flag_whole_protein].squeeze(-2) 
            
            
            compound_center_pred = scatter_mean(src=compound_coords_out_whole_protein, index=compound_batch, dim=0).unsqueeze(-2) 
            protein_coords_batched, protein_coords_batched_mask = to_dense_batch(data.protein_node_xyz, protein_batch) 
            
            protein_com_center_dis_map_pred = torch.cdist(protein_coords_batched, compound_center_pred) 
            protein_com_center_dis_map_pred=protein_com_center_dis_map_pred[protein_coords_batched_mask]
            protein_com_center_dis_map_pred = torch.clamp(protein_com_center_dis_map_pred, 0, 10).squeeze(-1) 
            
            compound_center_true=data.compound_coords_mean.unsqueeze(-2)
            protein_com_center_dis_map_true = torch.cdist(protein_coords_batched, compound_center_true) 
            protein_com_center_dis_map_true=protein_com_center_dis_map_true[protein_coords_batched_mask]
            protein_com_center_dis_map_true = torch.clamp(protein_com_center_dis_map_true, 0, 10).squeeze(-1) 
    
            return compound_center_pred.squeeze(-2),compound_center_true.squeeze(-2),protein_com_center_dis_map_pred,protein_com_center_dis_map_true   
                 
        elif self.args.location_pred_type=='pointnet_EGNN':
            compound_batch = data['compound'].batch
            compound_global_batch=data['compound_global_feat'].batch
            protein_batch = data['protein'].batch
            compound_rdkit_center_whole_protein_batch = data['compound_rdkit_center_whole_protein'].batch

            batched_compound_rdkit_center_whole_protein = self.normalize_coord(data['compound_rdkit_center_whole_protein'].node_coords.unsqueeze(-2)) 
            
            all_compound_feat_batched,all_compound_feat_mask=to_dense_batch(data['compound'].node_feats, compound_batch) 
            all_compound_feat_batched_trans=all_compound_feat_batched.transpose(2, 1)
            global_pro_esm_feat=self.compound_global_feat_pointnet(all_compound_feat_batched_trans,all_compound_feat_mask) 
                
            batched_compound_center_global_feat = self.location_compound_input_linear(global_pro_esm_feat) 
            batched_protein_emb_whole_protein = self.location_protein_input_linear(data['protein'].node_feats) 

            for i in range(compound_rdkit_center_whole_protein_batch.max()+1):
                if i == 0:
                    new_samples_whole_protein = torch.cat((
                        self.glb_c, batched_compound_center_global_feat[compound_global_batch==i], 
                        self.glb_p, batched_protein_emb_whole_protein[protein_batch==i]
                        ), dim=0)
                else:
                    new_sample_whole_protein = torch.cat((
                        self.glb_c, batched_compound_center_global_feat[compound_global_batch==i], 
                        self.glb_p, batched_protein_emb_whole_protein[protein_batch==i]
                        ), dim=0)
                    new_samples_whole_protein = torch.cat((new_samples_whole_protein, new_sample_whole_protein), dim=0)


            complex_center_coords_whole_protein, complex_center_out_whole_protein = self.location_pred_model(
                batched_compound_rdkit_center_whole_protein,
                new_samples_whole_protein, 
                batch_id=compound_rdkit_center_whole_protein_batch, 
                segment_id=data['compound_rdkit_center_whole_protein'].segment,
                mask=data['compound_rdkit_center_whole_protein'].mask, 
                is_global=data['compound_rdkit_center_whole_protein'].is_global,
                compound_edge_index=data['compound_rdkit_center_whole_protein', 'c2c', 'compound_rdkit_center_whole_protein'].edge_index,
                LAS_edge_index=None,
                batched_complex_coord_LAS=None,
                LAS_mask=None
            )
            
            complex_center_coords_whole_protein=self.unnormalize_coord(complex_center_coords_whole_protein)
            compound_center_flag_whole_protein = torch.logical_and(data['compound_rdkit_center_whole_protein'].segment == 0, ~data['compound_rdkit_center_whole_protein'].is_global) 
            compound_center_pred = complex_center_coords_whole_protein[compound_center_flag_whole_protein]
            
            protein_coords_batched, protein_coords_batched_mask = to_dense_batch(data.protein_node_xyz, protein_batch) 
            
            protein_com_center_dis_map_pred = torch.cdist(protein_coords_batched, compound_center_pred) 
            protein_com_center_dis_map_pred=protein_com_center_dis_map_pred[protein_coords_batched_mask]
            protein_com_center_dis_map_pred = torch.clamp(protein_com_center_dis_map_pred, 0, 10).squeeze(-1) 
            
            compound_center_true=data.compound_coords_mean.unsqueeze(-2)
            protein_com_center_dis_map_true = torch.cdist(protein_coords_batched, compound_center_true) 
            protein_com_center_dis_map_true=protein_com_center_dis_map_true[protein_coords_batched_mask]
            protein_com_center_dis_map_true = torch.clamp(protein_com_center_dis_map_true, 0, 10).squeeze(-1) 
    
            return compound_center_pred.squeeze(-2),compound_center_true.squeeze(-2),protein_com_center_dis_map_pred,protein_com_center_dis_map_true
        
        elif self.args.location_pred_type=='EGNN_pointnet':
            compound_batch = data['compound'].batch
            protein_batch = data['protein'].batch
            complex_batch_whole_protein = data['complex_whole_protein'].batch

            batched_complex_coord_whole_protein = self.normalize_coord(data['complex_whole_protein'].node_coords.unsqueeze(-2)) 
            batched_complex_coord_LAS_whole_protein = self.normalize_coord(data['complex_whole_protein'].node_coords_LAS.unsqueeze(-2)) 
            batched_compound_emb_whole_protein = self.location_compound_input_linear(data['compound'].node_feats) 
            batched_protein_emb_whole_protein = self.location_protein_input_linear(data['protein'].node_feats) 

            for i in range(complex_batch_whole_protein.max()+1):
                if i == 0:
                    new_samples_whole_protein = torch.cat((
                        self.glb_c, batched_compound_emb_whole_protein[compound_batch==i], 
                        self.glb_p, batched_protein_emb_whole_protein[protein_batch==i]
                        ), dim=0)
                else:
                    new_sample_whole_protein = torch.cat((
                        self.glb_c, batched_compound_emb_whole_protein[compound_batch==i], 
                        self.glb_p, batched_protein_emb_whole_protein[protein_batch==i]
                        ), dim=0)
                    new_samples_whole_protein = torch.cat((new_samples_whole_protein, new_sample_whole_protein), dim=0)

            complex_coords_whole_protein, complex_out_whole_protein = self.location_pred_model(
                batched_complex_coord_whole_protein,
                new_samples_whole_protein, 
                batch_id=complex_batch_whole_protein, 
                segment_id=data['complex_whole_protein'].segment,
                mask=data['complex_whole_protein'].mask, 
                is_global=data['complex_whole_protein'].is_global,
                compound_edge_index=data['complex_whole_protein', 'c2c', 'complex_whole_protein'].edge_index,
                LAS_edge_index=data['complex_whole_protein', 'LAS', 'complex_whole_protein'].edge_index,
                batched_complex_coord_LAS=batched_complex_coord_LAS_whole_protein,
                LAS_mask=None
            )
            
            complex_coords_whole_protein=self.unnormalize_coord(complex_coords_whole_protein)
            
            compound_flag_whole_protein = torch.logical_and(data['complex_whole_protein'].segment == 0, ~data['complex_whole_protein'].is_global) 
            compound_coords_out_whole_protein = complex_coords_whole_protein[compound_flag_whole_protein].squeeze(-2) 
            compound_center_pred_mean = scatter_mean(src=compound_coords_out_whole_protein, index=compound_batch, dim=0).unsqueeze(-2) 


            new_compound_coords_out_batched,new_compound_coords_out_mask=to_dense_batch(compound_coords_out_whole_protein, compound_batch) 
            new_compound_coords_out_batched_trans=new_compound_coords_out_batched.transpose(2, 1)
            global_new_compound_coords=self.compound_global_crood_pointnet(new_compound_coords_out_batched_trans,new_compound_coords_out_mask) 
            compound_center_pred_pointnet=self.compound_coord_center_linear(global_new_compound_coords).unsqueeze(-2)         
            
            compound_center_pred=0.8*compound_center_pred_mean+0.2*compound_center_pred_pointnet
            
            protein_coords_batched, protein_coords_batched_mask = to_dense_batch(data.protein_node_xyz, protein_batch) 
            
            protein_com_center_dis_map_pred = torch.cdist(protein_coords_batched, compound_center_pred) 
            protein_com_center_dis_map_pred=protein_com_center_dis_map_pred[protein_coords_batched_mask]
            protein_com_center_dis_map_pred = torch.clamp(protein_com_center_dis_map_pred, 0, 20).squeeze(-1) 
            
            compound_center_true=data.compound_coords_mean.unsqueeze(-2)
            protein_com_center_dis_map_true = torch.cdist(protein_coords_batched, compound_center_true) 
            protein_com_center_dis_map_true=protein_com_center_dis_map_true[protein_coords_batched_mask]
            protein_com_center_dis_map_true = torch.clamp(protein_com_center_dis_map_true, 0, 20).squeeze(-1)
    
            return compound_center_pred.squeeze(-2),compound_center_true.squeeze(-2),protein_com_center_dis_map_pred,protein_com_center_dis_map_true,compound_center_pred_pointnet.squeeze(-2)
        
def get_model(args):
    model = model_FABind_layer(args)
    return model
