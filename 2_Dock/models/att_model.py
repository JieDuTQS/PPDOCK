import torch
from torch import masked_fill, nn
import torch.nn.functional as F
import random
from torch_scatter import scatter_sum
from torch_geometric.utils import to_dense_batch
from models.egnn import MCAttEGNN
from models.model_utils import InteractionModule

def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res

def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res

class ComplexGraph(nn.Module):
    def __init__(self, args, inter_cutoff=10, intra_cutoff=8, normalize_coord=None, unnormalize_coord=None):
        super().__init__()
        self.args = args
        self.inter_cutoff = normalize_coord(inter_cutoff)
        self.intra_cutoff = normalize_coord(intra_cutoff)

    @torch.no_grad()
    def construct_edges(self, X, batch_id, segment_ids, is_global):
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id) 
        N, max_n = batch_id.shape[0], torch.max(lengths)
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)  
        gni = torch.arange(N, device=batch_id.device)
        gni2lni = gni - offsets[batch_id] 
        ctx_edges, inter_edges = [], []
        same_bid = torch.zeros(N, max_n, device=batch_id.device)
        same_bid[(gni, lengths[batch_id] - 1)] = 1
        same_bid = 1 - torch.cumsum(same_bid, dim=-1)
        same_bid = F.pad(same_bid[:, :-1], pad=(1, 0), value=1)
        same_bid[(gni, gni2lni)] = 0  
        row, col = torch.nonzero(same_bid).T  
        col = col + offsets[batch_id[row]] 
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))
        row_seg, col_seg = segment_ids[row], segment_ids[col]
        select_edges = sequential_and(
            row_seg == col_seg, 
            row_seg == 1,
            not_global_edges
        )
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]
        ctx_edges = _radial_edges(X, torch.stack([ctx_all_row, ctx_all_col]).T, cutoff=self.intra_cutoff)
        select_edges = torch.logical_and(row_seg != col_seg, not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        inter_edges = _radial_edges(X, torch.stack([inter_all_row, inter_all_col]).T, cutoff=self.inter_cutoff)
        if inter_edges.shape[1] == 0:
            inter_edges = torch.tensor([[inter_all_row[0], inter_all_col[0]], [inter_all_col[0], inter_all_row[0]]], device=inter_all_row.device)
        reduced_inter_edge_batchid = batch_id[inter_edges[0][inter_edges[0] < inter_edges[1]]] 
        reduced_inter_edge_offsets = offsets.gather(-1, reduced_inter_edge_batchid)
        select_edges = torch.logical_and(row_seg == col_seg, torch.logical_not(not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]]) 
        select_edges = torch.logical_and(row_global, col_global) 
        global_global = torch.stack([row[select_edges], col[select_edges]])  
        space_edge_num = ctx_edges.shape[1] + global_normal.shape[1] + global_global.shape[1]
        ctx_edges = torch.cat([ctx_edges, global_normal, global_global], dim=1) 
        if self.args.add_attn_pair_bias:
            return ctx_edges, inter_edges, (reduced_inter_edge_batchid, reduced_inter_edge_offsets)
        else:
            return ctx_edges, inter_edges, None
    def forward(self, X, batch_id, segment_id, is_global):
        return self.construct_edges(X, batch_id, segment_id, is_global)

def _radial_edges(X, src_dst, cutoff):
    dist = X[:, 0][src_dst] 
    dist = torch.norm(dist[:, 0] - dist[:, 1], dim=-1) 
    src_dst = src_dst[dist <= cutoff]
    src_dst = src_dst.transpose(0, 1) 
    return src_dst

class EfficientMCAttModel(nn.Module):
    def __init__(self, args=None, embed_size=512, hidden_size=512, n_channel=1, n_edge_feats=0,
                 n_layers=5, dropout=0.1, n_iter=8, dense=False, 
                 inter_cutoff=10, intra_cutoff=8, normalize_coord=None, unnormalize_coord=None):
        super().__init__()
        self.n_iter = n_iter
        self.args = args
        self.random_n_iter = args.random_n_iter
        self.gnn = MCAttEGNN(args, embed_size, hidden_size, hidden_size,
                            n_channel, n_edge_feats, n_layers=n_layers,
                            residual=True, dropout=dropout, dense=dense,
                            normalize_coord=normalize_coord, unnormalize_coord=unnormalize_coord,
                            geometry_reg_step_size=args.geometry_reg_step_size)
        self.extract_edges = ComplexGraph(args, inter_cutoff=inter_cutoff, intra_cutoff=intra_cutoff, normalize_coord=normalize_coord, unnormalize_coord=unnormalize_coord)
        self.inter_layer = InteractionModule(hidden_size, hidden_size, hidden_size, rm_layernorm=args.rm_layernorm)
    def forward(self, X, H, batch_id, segment_id, mask, is_global, compound_edge_index, LAS_edge_index, batched_complex_coord_LAS, LAS_mask=None):
        p_p_dist_embed=None
        c_c_dist_embed=None
        c_batch = batch_id[segment_id == 0]
        p_batch = batch_id[segment_id == 1]
        c_embed = H[segment_id == 0]
        p_embed = H[segment_id == 1]
        p_embed_batched, p_mask = to_dense_batch(p_embed, p_batch)
        c_embed_batched, c_mask = to_dense_batch(c_embed, c_batch)
        pair_embed_batched, pair_mask = self.inter_layer(p_embed_batched, c_embed_batched, p_mask, c_mask)
        pair_embed_batched = pair_embed_batched * pair_mask.to(torch.float).unsqueeze(-1)
        if self.training and self.random_n_iter:
            iter_i = random.randint(1, self.n_iter)
        else:
            iter_i = self.n_iter
        for r in range(iter_i):
            if r < iter_i - 1:
                with torch.no_grad():
                    ctx_edges, inter_edges, reduced_tuple = self.extract_edges(X, batch_id, segment_id, is_global)
                    ctx_edges = torch.cat((compound_edge_index, ctx_edges), dim=1)
                    _, Z = self.gnn(H, X, ctx_edges, inter_edges, LAS_edge_index, batched_complex_coord_LAS,
                                    segment_id=segment_id, batch_id=batch_id, reduced_tuple=reduced_tuple,
                                    pair_embed_batched=pair_embed_batched, pair_mask=pair_mask, LAS_mask=LAS_mask,
                                    p_p_dist_embed=p_p_dist_embed, c_c_dist_embed=c_c_dist_embed, mask=mask)
                    X[mask] = Z[mask]
            else:
                with torch.no_grad():
                    ctx_edges, inter_edges, reduced_tuple = self.extract_edges(X, batch_id, segment_id, is_global)
                    ctx_edges = torch.cat((compound_edge_index, ctx_edges), dim=1)
                H, Z = self.gnn(H, X, ctx_edges, inter_edges, LAS_edge_index, batched_complex_coord_LAS,
                                segment_id=segment_id, batch_id=batch_id, reduced_tuple=reduced_tuple,
                                pair_embed_batched=pair_embed_batched, pair_mask=pair_mask, LAS_mask=LAS_mask,
                                p_p_dist_embed=p_p_dist_embed, c_c_dist_embed=c_c_dist_embed, mask=mask)
                X[mask] = Z[mask]
        return X, H
    

