import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_softmax, scatter_add, scatter_sum
from torch_geometric.utils import to_dense_batch, to_dense_adj
from models.Cross_Attention_Model import CrossAttentionModule
from models.model_utils import InteractionModule, RBFDistanceModule



class MC_E_GCL(nn.Module):
    def __init__(self, args, input_nf, output_nf, hidden_nf, n_channel, edges_in_d=0, act_fn=nn.SiLU(),
                 residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False,
                 dropout=0.1, coord_change_maximum=10):
        super(MC_E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.args = args
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8

        self.dropout = nn.Dropout(dropout)

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + n_channel**2 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, n_channel, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())
        self.coord_change_maximum = coord_change_maximum

    def edge_model(self, source, target, radial, edge_attr):
        radial = radial.reshape(radial.shape[0], -1)  

        if edge_attr is None:  
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        out = self.dropout(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))

        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)  

        out = self.node_mlp(agg)  

        out = self.dropout(out)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat).unsqueeze(-1) 

        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))  
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg.clamp(-self.coord_change_maximum, self.coord_change_maximum)
        return coord

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, batch_id=None):
        row, col = edge_index
        radial, coord_diff = coord2radial(edge_index, coord, self.args.rm_F_norm, batch_id=batch_id, norm_type=self.args.norm_type)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr) 

        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)    
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord


class MC_Att_L(nn.Module):
    def __init__(self, args, input_nf, output_nf, hidden_nf, n_channel, edges_in_d=0,
                 act_fn=nn.SiLU(), dropout=0.1, coord_change_maximum=10, opm=False, normalize_coord=None):
        super().__init__()
        self.args = args

        self.hidden_nf = hidden_nf

        self.dropout = nn.Dropout(dropout)

        self.linear_q = nn.Linear(input_nf, hidden_nf)
        self.linear_kv = nn.Linear(input_nf + n_channel ** 2 + edges_in_d, hidden_nf * 2)  

        layer = nn.Linear(hidden_nf, n_channel, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        self.coord_mlp = nn.Sequential(*coord_mlp)
        self.coord_change_maximum = coord_change_maximum

        if args.add_cross_attn_layer and args.explicit_pair_embed:
            self.cross_attn_module = CrossAttentionModule(node_hidden_dim=input_nf, pair_hidden_dim=input_nf, rm_layernorm=args.rm_layernorm, keep_trig_attn=args.keep_trig_attn, dist_hidden_dim=input_nf, normalize_coord=normalize_coord)
        elif args.add_cross_attn_layer and not args.explicit_pair_embed:
            raise AssertionError
        elif args.add_cross_attn_layer and not args.add_attn_pair_bias:
            raise AssertionError

        if args.add_attn_pair_bias:
            self.inter_layer = InteractionModule(input_nf, output_nf, hidden_nf, opm=opm, rm_layernorm=args.rm_layernorm)
            self.attn_bias_proj = nn.Linear(hidden_nf, 1)
        


    def att_model(self, h, edge_index, radial, edge_attr, pair_embed=None):
        row, col = edge_index
        source, target = h[row], h[col] 

        q = self.linear_q(source)  
        n_channel = radial.shape[1]
        radial = radial.reshape(radial.shape[0], n_channel * n_channel)  
        if edge_attr is not None:
            target_feat = torch.cat([radial, target, edge_attr], dim=1)
        else:
            target_feat = torch.cat([radial, target], dim=1)
        kv = self.linear_kv(target_feat)  
        k, v = kv[..., 0::2], kv[..., 1::2]  

        if self.args.add_attn_pair_bias:
            attn_bias = self.attn_bias_proj(pair_embed).squeeze(-1) 
            alpha = torch.sum(q * k, dim=1) + attn_bias 

        else:
            alpha = torch.sum(q * k, dim=1)  

        alpha = scatter_softmax(alpha, row) 
        return alpha, v
        
    def node_model(self, h, edge_index, att_weight, v):
        row, _ = edge_index
        agg = unsorted_segment_sum(att_weight * v, row, h.shape[0]) 
        agg = self.dropout(agg)
        return h + agg  

    def coord_model(self, coord, edge_index, coord_diff, att_weight, v):
        row, _ = edge_index
        coord_v = att_weight * self.coord_mlp(v) 
        trans = coord_diff * coord_v.unsqueeze(-1)
        agg = unsorted_segment_sum(trans, row, coord.size(0))
        coord = coord + agg.clamp(-self.coord_change_maximum, self.coord_change_maximum)
        return coord
    
    def trio_encoder(self, h, edge_index, coord, pair_embed_batched=None, pair_mask=None, batch_id=None, segment_id=None, reduced_tuple=None, LAS_mask=None, p_p_dist_embed=None, c_c_dist_embed=None):
        row, col = edge_index
        c_batch = batch_id[segment_id == 0]
        p_batch = batch_id[segment_id == 1]
        c_embed = h[segment_id == 0]
        p_embed = h[segment_id == 1]
        p_embed_batched, p_mask = to_dense_batch(p_embed, p_batch)
        c_embed_batched, c_mask = to_dense_batch(c_embed, c_batch)
        if self.args.add_cross_attn_layer:
            p_embed_batched, c_embed_batched, pair_embed_batched = self.cross_attn_module(
                p_embed_batched, p_mask,
                c_embed_batched, c_mask,
                pair_embed_batched, pair_mask,
                p_p_dist_embed=p_p_dist_embed, c_c_dist_embed=c_c_dist_embed
            )
            for i in range(batch_id.max()+1):
                if i == 0:
                    new_h = torch.cat((c_embed_batched[i][c_mask[i]], p_embed_batched[i][p_mask[i]]), dim=0)
                else:
                    new_sample = torch.cat((c_embed_batched[i][c_mask[i]], p_embed_batched[i][p_mask[i]]), dim=0)
                    new_h = torch.cat((new_h, new_sample), dim=0)
        else:
            new_h = h
            if self.args.explicit_pair_embed:
                pair_embed_batched = pair_embed_batched + self.inter_layer(p_embed_batched, c_embed_batched, p_mask, c_mask)[0]
            else:
                pair_embed_batched = self.inter_layer(p_embed_batched, c_embed_batched, p_mask, c_mask)[0]

        compound_offset_in_batch = c_mask.sum(1)
        reduced_inter_edges_batchid, reduced_inter_edge_offsets = reduced_tuple
        reduced_row = row[row<col] - reduced_inter_edge_offsets 
        reduced_col = col[row<col] - reduced_inter_edge_offsets -  compound_offset_in_batch[reduced_inter_edges_batchid]
        first_part = pair_embed_batched[reduced_inter_edges_batchid, reduced_col, reduced_row] 
        reduced_row = row[row>col] - reduced_inter_edge_offsets -  compound_offset_in_batch[reduced_inter_edges_batchid]
        reduced_col = col[row>col] - reduced_inter_edge_offsets
        second_part = pair_embed_batched[reduced_inter_edges_batchid, reduced_row, reduced_col] 
        for i in range(reduced_inter_edges_batchid.max()+1):
            if i == 0:
                pair_offset = torch.cat((
                    first_part[reduced_inter_edges_batchid==i], second_part[reduced_inter_edges_batchid==i]
                    ), dim=0)
            else:
                new_sample = torch.cat((
                    first_part[reduced_inter_edges_batchid==i], second_part[reduced_inter_edges_batchid==i]
                    ), dim=0)
                pair_offset = torch.cat((pair_offset, new_sample), dim=0)
        return new_h, pair_embed_batched, pair_offset


    def forward(self, h, edge_index, coord, edge_attr=None, segment_id=None, batch_id=None,
                 reduced_tuple=None, pair_embed_batched=None, pair_mask=None, LAS_mask=None,
                 p_p_dist_embed=None, c_c_dist_embed=None):
        if self.args.add_attn_pair_bias:
            h, pair_embed_batched, pair_offset_embed = self.trio_encoder(
                h, edge_index, coord, pair_embed_batched=pair_embed_batched, pair_mask=pair_mask, 
                batch_id=batch_id, segment_id=segment_id, reduced_tuple=reduced_tuple, LAS_mask=LAS_mask, 
                p_p_dist_embed=p_p_dist_embed, c_c_dist_embed=c_c_dist_embed
            )
        else:
            pair_offset_embed = None
        
        radial, coord_diff = coord2radial(edge_index, coord, self.args.rm_F_norm, batch_id=batch_id, norm_type=self.args.norm_type)
        att_weight, v = self.att_model(h, edge_index, radial, edge_attr, pair_embed=pair_offset_embed)

        flat_att_weight = att_weight
        att_weight = att_weight.unsqueeze(-1) 
        h = self.node_model(h, edge_index, att_weight, v)
        coord = self.coord_model(coord, edge_index, coord_diff, att_weight, v)

        return h, coord, flat_att_weight


class MCAttEGNN(nn.Module):
    def __init__(self, args, in_node_nf, hidden_nf, out_node_nf, n_channel, in_edge_nf=0,
                 act_fn=nn.SiLU(), n_layers=4, residual=True, dropout=0.1, dense=False,
                 normalize_coord=None, unnormalize_coord=None, geometry_reg_step_size=0.001):
        super().__init__()
        self.args = args

        self.geometry_reg_step_size = geometry_reg_step_size
        self.geom_reg_steps = 1
        
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout)

        self.linear_in = nn.Linear(in_node_nf, self.hidden_nf)

        self.dense = dense
        self.normalize_coord = normalize_coord
        self.unnormalize_coord = unnormalize_coord
        if dense:
            self.linear_out = nn.Linear(self.hidden_nf * (n_layers + 1), out_node_nf)
        else:
            self.linear_out = nn.Linear(self.hidden_nf, out_node_nf)

        for i in range(0, n_layers):
            self.add_module(f'gcl_{i}', MC_E_GCL(
                args, self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel,
                edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, dropout=dropout,
                coord_change_maximum=self.normalize_coord(10)
            ))

            self.add_module(f'att_{i}', MC_Att_L(
                args, self.hidden_nf, self.hidden_nf, self.hidden_nf,
                n_channel, edges_in_d=0, act_fn=act_fn, dropout=dropout,
                coord_change_maximum=self.normalize_coord(10), opm=args.opm, normalize_coord=normalize_coord
            ))
        self.out_layer = MC_E_GCL(
            args, self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel,
            edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual,
            coord_change_maximum=self.normalize_coord(10)
        )
    
    def forward(self, h, x, ctx_edges, att_edges, LAS_edge_list, batched_complex_coord_LAS, 
                segment_id=None, batch_id=None, reduced_tuple=None, 
                pair_embed_batched=None, pair_mask=None, LAS_mask=None,
                p_p_dist_embed=None, c_c_dist_embed=None, mask=None,
                ctx_edge_attr=None, att_edge_attr=None, return_attention=False):
        h = self.linear_in(h)
        h = self.dropout(h)
        x = x.clone()

        ctx_states, ctx_coords, atts = [], [], []
        for i in range(0, self.n_layers):
            h, coord = self._modules[f'gcl_{i}'](h, ctx_edges, x, batch_id=batch_id, edge_attr=ctx_edge_attr)
            if self.args.fix_pocket:
                x[mask] = coord[mask]
            else:
                x = coord
            ctx_states.append(h)
            ctx_coords.append(x)

            if self.args.add_attn_pair_bias:
                if self.args.explicit_pair_embed:
                    h, coord, att = self._modules[f'att_{i}'](
                        h, att_edges, x, edge_attr=att_edge_attr, 
                        segment_id=segment_id, batch_id=batch_id, reduced_tuple=reduced_tuple,
                        pair_embed_batched=pair_embed_batched, pair_mask=pair_mask, LAS_mask=LAS_mask, 
                        p_p_dist_embed=p_p_dist_embed, c_c_dist_embed=c_c_dist_embed
                        )
                else:
                    h, coord, att = self._modules[f'att_{i}'](
                        h, att_edges, x, edge_attr=att_edge_attr, 
                        segment_id=segment_id, batch_id=batch_id, reduced_tuple=reduced_tuple
                        )
            else:
                h, coord, att = self._modules[f'att_{i}'](h, att_edges, x, batch_id=batch_id, edge_attr=att_edge_attr)

            if self.args.fix_pocket:
                x[mask] = coord[mask]
            else:
                x = coord
            atts.append(att)

            if not self.args.rm_LAS_constrained_optim:
                x.squeeze_(1)
                batched_complex_coord_LAS.squeeze_(1)
                for step in range(self.geom_reg_steps):
                    LAS_cur_squared = torch.sum(
                        (x[LAS_edge_list[0]] - x[LAS_edge_list[1]]) ** 2, dim=1)
                    LAS_true_squared = torch.sum(
                        (batched_complex_coord_LAS[LAS_edge_list[0]] - batched_complex_coord_LAS[LAS_edge_list[1]]) ** 2,
                        dim=1)
                    grad_squared = 2 * (x[LAS_edge_list[0]] - x[LAS_edge_list[1]])
                    LAS_force = 2 * (LAS_cur_squared - LAS_true_squared)[:, None] * grad_squared
                    LAS_delta_coord = scatter_add(src=LAS_force, index=LAS_edge_list[1], dim=0,
                                                dim_size=x.shape[0])

                    x = x + (LAS_delta_coord * self.geometry_reg_step_size) \
                        .clamp(min=self.normalize_coord(-15), max=self.normalize_coord(15))
                x.unsqueeze_(1)

        h, coord = self.out_layer(h, ctx_edges, x, batch_id=batch_id, edge_attr=ctx_edge_attr)
        if self.args.fix_pocket:
            x[mask] = coord[mask]
        else:
            x = coord
        ctx_states.append(h)
        ctx_coords.append(x)
        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
        h = self.dropout(h)
        h = self.linear_out(h)
        if return_attention:
            return h, x, atts
        else:
            return h, x



def coord2radial(edge_index, coord, rm_F_norm, batch_id=None, norm_type=None):
    row, col = edge_index
    coord_diff = coord[row] - coord[col] 
    radial = torch.bmm(coord_diff, coord_diff.transpose(-1, -2))  
 
    if not rm_F_norm:
        if norm_type == 'all_sample':
            radial = F.normalize(radial, dim=0) 
        elif norm_type == 'per_sample':
            edge_batch_id = batch_id[row]
            norm_for_each_sample = scatter_sum(src=(radial**2), index=edge_batch_id, dim=0).sqrt()
            norm_for_each_edge = norm_for_each_sample[edge_batch_id]
            radial = radial / norm_for_each_edge
        elif norm_type == '4_sample':
            shrink_batch_id = batch_id // 4
            edge_batch_id = shrink_batch_id[row]
            norm_for_each_sample = scatter_sum(src=(radial**2), index=edge_batch_id, dim=0).sqrt()
            norm_for_each_edge = norm_for_each_sample[edge_batch_id]
            radial = radial / norm_for_each_edge

    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments):
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


