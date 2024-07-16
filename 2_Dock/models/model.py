import torch
from torch_geometric.utils import to_dense_batch
from torch import nn
from models.att_model import EfficientMCAttModel

class model_FABind_layer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding_channels=args.hidden_size
        self.layernorm = torch.nn.LayerNorm(self.embedding_channels)
        self.coordinate_scale = args.coordinate_scale
        self.normalize_coord = lambda x: x / self.coordinate_scale 
        self.unnormalize_coord = lambda x: x * self.coordinate_scale 
        n_channel = 1 
        self.complex_binding_model = EfficientMCAttModel(
            args, self.embedding_channels, self.embedding_channels, n_channel, n_edge_feats=0, n_layers=args.num_layers, n_iter=args.num_iters,
            inter_cutoff=args.inter_cutoff, intra_cutoff=args.intra_cutoff, normalize_coord=self.normalize_coord, unnormalize_coord=self.unnormalize_coord)
        self.glb_c = nn.Parameter(torch.ones(1, self.embedding_channels))
        self.glb_p = nn.Parameter(torch.ones(1, self.embedding_channels))
        protein_feat_size = args.protein_feat_size
        compound_feat_size = args.compound_feat_size
        self.protein_input_linear = nn.Linear(protein_feat_size, self.embedding_channels)  
        self.compound_input_linear = nn.Linear(compound_feat_size, self.embedding_channels) 
        self.distmap_mlp = nn.Sequential(
            nn.Linear(self.embedding_channels, self.embedding_channels),
            nn.ReLU(),
            nn.Linear(self.embedding_channels, 1))
        torch.nn.init.xavier_uniform_(self.protein_input_linear.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.compound_input_linear.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.distmap_mlp[0].weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.distmap_mlp[2].weight, gain=0.001)

    def forward(self, data):
        compound_batch = data['compound'].batch
        pocket_batch = data['pocket'].batch
        complex_batch = data['complex'].batch
        batched_compound_emb = self.compound_input_linear(data['compound'].node_feats) 
        batched_pocket_emb = self.protein_input_linear(data['pocket'].node_feats) 
        batched_complex_coord = self.normalize_coord(data['complex'].node_coords.unsqueeze(-2)) 
        batched_complex_coord_LAS = self.normalize_coord(data['complex'].node_coords_LAS.unsqueeze(-2)) 
        for i in range(complex_batch.max()+1): 
            if i == 0:
                batched_complex_emb = torch.cat((
                    self.glb_c, batched_compound_emb[compound_batch==i], 
                    self.glb_p, batched_pocket_emb[pocket_batch==i]
                    ), dim=0)
            else:
                complex_emb = torch.cat((
                    self.glb_c, batched_compound_emb[compound_batch==i], 
                    self.glb_p, batched_pocket_emb[pocket_batch==i]
                    ), dim=0)
                batched_complex_emb = torch.cat((batched_complex_emb, complex_emb), dim=0)
        dis_map = data.dis_map_poc_com
        batched_complex_coord_out, batched_complex_emb_out = self.complex_binding_model(
            batched_complex_coord,
            batched_complex_emb, 
            batch_id=complex_batch, 
            segment_id=data['complex'].segment,
            mask=data['complex'].mask, 
            is_global=data['complex'].is_global,
            compound_edge_index=data['complex', 'c2c', 'complex'].edge_index,
            LAS_edge_index=data['complex', 'LAS', 'complex'].edge_index,
            batched_complex_coord_LAS=batched_complex_coord_LAS,
            LAS_mask=None
        )
        pocket_coords_batched, _ = to_dense_batch(self.normalize_coord(data.pocket_node_xyz), pocket_batch) 
        compound_flag = torch.logical_and(data['complex'].segment == 0, ~data['complex'].is_global)
        pocket_flag  = torch.logical_and(data['complex'].segment == 1, ~data['complex'].is_global)
        compound_emb_out = batched_complex_emb_out[compound_flag] 
        compound_coords_out = batched_complex_coord_out[compound_flag].squeeze(-2) 
        pocket_emb_out  = batched_complex_emb_out[pocket_flag] 
        compound_emb_out_batched, compound_out_mask_batched = to_dense_batch(compound_emb_out, compound_batch) 
        compound_coords_out_batched, compound_coords_out_mask_batched = to_dense_batch(compound_coords_out, compound_batch) 
        pocket_emb_out_batched, pocket_out_mask_batched = to_dense_batch(pocket_emb_out, pocket_batch) 
        pocket_com_dis_map = torch.cdist(pocket_coords_batched, compound_coords_out_batched) 
        pocket_emb_out_batched = self.layernorm(pocket_emb_out_batched) 
        compound_emb_out_batched = self.layernorm(compound_emb_out_batched) 
        z = torch.einsum("bik,bjk->bijk", pocket_emb_out_batched, compound_emb_out_batched) 
        z_mask = torch.einsum("bi,bj->bij", pocket_out_mask_batched, compound_out_mask_batched) 
        b = self.distmap_mlp(z).squeeze(-1)
        dis_map_pred_by_pair_embeddings = b[z_mask] 
        dis_map_pred_by_pair_embeddings = dis_map_pred_by_pair_embeddings.sigmoid() * 10 
        dis_map_pred_by_coord = pocket_com_dis_map[z_mask] 
        dis_map_pred_by_coord = self.unnormalize_coord(dis_map_pred_by_coord) 
        dis_map_pred_by_coord = torch.clamp(dis_map_pred_by_coord, 0, 10) 
        compound_coord_pred = self.unnormalize_coord(compound_coords_out)
        return compound_coord_pred, compound_batch, dis_map_pred_by_pair_embeddings, dis_map_pred_by_coord, dis_map

def get_model(args, logger, device):
    if args.model_net == "FABind_layer":
        model = model_FABind_layer(args)
    return model
