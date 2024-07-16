import math
from typing import List, Tuple, Optional

import torch
from torch.nn import Linear, LayerNorm
from torch.nn.functional import softmax
import torch.nn as nn



def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, biases: List[torch.Tensor]) -> torch.Tensor:
    query = permute_final_dims(query, (1, 0, 2))
    key = permute_final_dims(key, (1, 2, 0))
    value = permute_final_dims(value, (1, 0, 2))
    a = torch.matmul(query, key)
    for b in biases:
        a = a + b
    a = softmax(a, -1)
    a = torch.matmul(a, value)
    a = a.transpose(-2, -3)

    return a


class Attention(nn.Module):
    def __init__(
            self,
            c_q: int,
            c_k: int,
            c_v: int,
            c_hidden: int,
            no_heads: int,
            gating: bool = True,
    ):

        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = Linear(self.c_q, self.c_hidden * self.no_heads, bias=False)
        self.linear_k = Linear(self.c_k, self.c_hidden * self.no_heads, bias=False)
        self.linear_v = Linear(self.c_v, self.c_hidden * self.no_heads, bias=False)
        self.linear_o = Linear(self.c_hidden * self.no_heads, self.c_q)

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self,
                  q_x: torch.Tensor,
                  kv_x: torch.Tensor
                  ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
     
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q /= math.sqrt(self.c_hidden)
 
        return q, k, v

    def _wrap_up(self,
                 o: torch.Tensor,
                 q_x: torch.Tensor
                 ) -> torch.Tensor:
        if (self.linear_g is not None):
            g = self.sigmoid(self.linear_g(q_x))

            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        o = flatten_final_dims(o, 2)

        o = self.linear_o(o)

        return o

    def forward(
            self,
            q_x: torch.Tensor,
            kv_x: torch.Tensor,
            biases: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:

        if biases is None:
            biases = []

        q, k, v = self._prep_qkv(q_x, kv_x)
        o = _attention(q, k, v, biases)
        o = self._wrap_up(o, q_x)

        return o


class Transition(torch.nn.Module):
    def __init__(self, hidden_dim=128, n=4, rm_layernorm=False):
        super().__init__()
        self.rm_layernorm = rm_layernorm
        if not self.rm_layernorm:        
            self.layernorm = torch.nn.LayerNorm(hidden_dim)
        self.linear_1 = Linear(hidden_dim, n * hidden_dim)
        self.linear_2 = Linear(n * hidden_dim, hidden_dim)

    def forward(self, x):
        if not self.rm_layernorm:    
            x = self.layernorm(x)
        x = self.linear_2((self.linear_1(x)).relu())
        return x

class InteractionModule(torch.nn.Module):
    def __init__(self, node_hidden_dim, pair_hidden_dim, hidden_dim, opm=False, rm_layernorm=False):
        super(InteractionModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.pair_hidden_dim = pair_hidden_dim
        self.node_hidden_dim = node_hidden_dim
        self.opm = opm

        self.rm_layernorm = rm_layernorm
        if not rm_layernorm:
            self.layer_norm_p = nn.LayerNorm(node_hidden_dim)
            self.layer_norm_c = nn.LayerNorm(node_hidden_dim)

        if self.opm:
            self.linear_p = nn.Linear(node_hidden_dim, hidden_dim)
            self.linear_c = nn.Linear(node_hidden_dim, hidden_dim)
            self.linear_out = nn.Linear(hidden_dim ** 2, pair_hidden_dim)
        else:
            self.linear_p = nn.Linear(node_hidden_dim, hidden_dim)
            self.linear_c = nn.Linear(node_hidden_dim, hidden_dim)
            self.linear_out = nn.Linear(hidden_dim, pair_hidden_dim)

    def forward(self, p_embed, c_embed,
                p_mask=None, c_mask=None):

        if p_mask is None:
            p_mask = p_embed.new_ones(p_embed.shape[:-1], dtype=torch.bool)
        if c_mask is None:
            c_mask = c_embed.new_ones(c_embed.shape[:-1], dtype=torch.bool)
        inter_mask = torch.einsum("...i,...j->...ij", p_mask, c_mask) 

        if not self.rm_layernorm:
            p_embed = self.layer_norm_p(p_embed)  
            c_embed = self.layer_norm_c(c_embed) 
        if self.opm:
            p_embed = self.linear_p(p_embed) 
            c_embed = self.linear_c(c_embed)  
            inter_embed = torch.einsum("...bc,...de->...bdce", p_embed, c_embed)
            inter_embed = torch.flatten(inter_embed, -2) 
            inter_embed = self.linear_out(inter_embed) * inter_mask.unsqueeze(-1)
        else:
            p_embed = self.linear_p(p_embed) 
            c_embed = self.linear_c(c_embed)  
            inter_embed = torch.einsum("...ik,...jk->...ijk", p_embed, c_embed)
            inter_embed = self.linear_out(inter_embed) * inter_mask.unsqueeze(-1)
        return inter_embed, inter_mask



class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist[..., None] - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class RBFDistanceModule(torch.nn.Module):
    def __init__(self, rbf_stop, distance_hidden_dim, num_gaussian=32, dropout=0.1):
        super(RBFDistanceModule, self).__init__()
        self.distance_hidden_dim = distance_hidden_dim
        self.rbf = GaussianSmearing(start=0, stop=rbf_stop, num_gaussians=num_gaussian)
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussian, distance_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(distance_hidden_dim, distance_hidden_dim)
        )

    def forward(self, distance):
        return self.mlp(self.rbf(distance))  