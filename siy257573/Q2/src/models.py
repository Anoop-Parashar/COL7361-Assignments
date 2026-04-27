"""
models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TAGConv, ARMAConv


# important
def _row_normalize(x):
    row_sum = x.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return x / row_sum


# Paper-exact models for Dataset A ensemble


class GATModelPaper(nn.Module):
    """
    Exact 2-layer GAT from Velickovic et al. ICLR 2018.
    Layer 1: 8 heads x 8 features (64 total), ELU activation.
    Layer 2: 1 head x num_classes.
    dropout=0.6 on inputs and attention. wd=5e-4.
    """

    def __init__(self, in_channels, out_channels, dropout=0.6):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=dropout, concat=True)
        self.conv2 = GATConv(64, out_channels, heads=1, dropout=dropout, concat=False)

    def forward(self, x, edge_index):
        x = _row_normalize(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class TAGModelPaper(nn.Module):
    """
    Exact 2-layer TAGCN from Du et al. ICLR 2019.
    2 hidden layers, 16 units each, K=3 filter, ReLU, dropout=0.5.
    """

    def __init__(self, in_channels, out_channels, dropout=0.5, K=3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = TAGConv(in_channels, 16, K=K)
        self.conv2 = TAGConv(16, out_channels, K=K)

    def forward(self, x, edge_index):
        x = _row_normalize(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class ARMAModelPaper(nn.Module):
    """
    Exact 2-layer ARMA from Bianchi et al. 2021, Table 6 Cora settings.
    hidden=16, K=2 stacks, T=1 depth, dropout=0.75, wd=5e-4.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden=16,
        dropout=0.75,
        num_stacks=2,
        num_layers=1,
    ):
        super().__init__()
        self.dropout = dropout
        self.conv1 = ARMAConv(
            in_channels,
            hidden,
            num_stacks=num_stacks,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.conv2 = ARMAConv(
            hidden, out_channels, num_stacks=1, num_layers=1, dropout=dropout
        )

    def forward(self, x, edge_index):
        x = _row_normalize(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


# Ensemble
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, edge_index):
        return torch.stack([m(x, edge_index) for m in self.models], dim=0).mean(dim=0)


# Dataset B


class SAGEModelB(nn.Module):
    def __init__(self, in_channels, hidden=256, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv3(x, edge_index)


# Dynamic Binning Embedding (DyBEM)


class DyBEMLayer(nn.Module):
    def __init__(self, in_dim, num_bins=10, embed_dim=64):
        super().__init__()
        # learnable bin boundaries
        self.bin_logits = nn.Parameter(torch.randn(num_bins))
        # embedding for each bin
        self.bin_embeds = nn.Embedding(num_bins, embed_dim)
        self.lin = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # normalize to [0,1]
        x_norm = (x - x.min(dim=0)[0]) / (x.max(dim=0)[0] - x.min(dim=0)[0] + 1e-6)
        # softmax to get bin probabilities
        probs = torch.softmax(self.bin_logits, dim=0)
        bins = torch.cumsum(probs, dim=0)
        # assign each feature to a bin index
        # idx = torch.bucketize(x_norm, bins)
        idx = torch.bucketize(x_norm, bins).clamp(0, self.bin_embeds.num_embeddings - 1)
        # embed bins
        out = self.bin_embeds(idx)
        return self.lin(out).sum(dim=1)  # aggregate across features


# Global Attention Aggregation
class GlobalAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.Wq = nn.Linear(in_dim, hidden_dim)
        self.Wk = nn.Linear(in_dim, hidden_dim)
        self.Wv = nn.Linear(in_dim, hidden_dim)

    def forward(self, batch_embeds, global_embeds):
        Q = self.Wq(batch_embeds)  # [B, H]
        K = self.Wk(global_embeds)  # [N, H]
        V = self.Wv(global_embeds)  # [N, H]

        attn = torch.softmax(Q @ K.T / (K.size(-1) ** 0.5), dim=-1)  # [B, N]
        out = attn @ V  # [B, H]
        return out


# GAAP Model
class GAAPModelB(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.3, num_bins=10):
        super().__init__()
        # self.dyBEM = DyBEMLayer(in_dim, num_bins=num_bins, embed_dim=64)
        self.input_proj = nn.Linear(in_dim, 64)
        self.sage1 = SAGEConv(64, hidden)
        self.sage2 = SAGEConv(hidden, hidden)
        self.global_attn = GlobalAttention(hidden, hidden)  # expects hidden-dim inputs
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, edge_index, global_cache=None, return_hidden=False):
        # DyBEM -> SAGE
        # x_emb = self.dyBEM(x)  # [N, embed_dim]
        x_emb = F.relu(self.input_proj(x))  # [N, 64]
        h = F.relu(self.sage1(x_emb, edge_index))  # [N, hidden]
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.sage2(h, edge_index)  # [N, hidden]

        # global attention expects global_cache of shape [M, hidden]
        if global_cache is not None:
            g = self.global_attn(h, global_cache)  # [N, hidden]
            h_cat = torch.cat([h, g], dim=1)  # [N, hidden*2]
        else:
            h_cat = torch.cat([h, h], dim=1)  # fallback

        out = self.mlp(h_cat)  # [N, 2]

        if return_hidden:
            return out, h  # return logits and the *hidden* h (not h_cat)
        return out


# Dataset C


# Structural feature helpers  (ALL CPU, sparse)
def _build_neighbor_sets(edge_index_cpu, num_nodes):
    neighbors = [set() for _ in range(num_nodes)]
    src = edge_index_cpu[0].tolist()
    dst = edge_index_cpu[1].tolist()
    for u, v in zip(src, dst):
        neighbors[u].add(v)
        neighbors[v].add(u)
    return neighbors


def _compute_spd_bfs(neighbors, src_list, dst_list, max_dist=5):
    """
    BFS shortest-path distance.
    Bug-fixed: the for-dist loop breaks immediately on found,
    so we never continue BFS after the target is reached.
    """
    results = []
    for u, v in zip(src_list, dst_list):
        if u == v:
            results.append(0)
            continue
        if v in neighbors[u]:
            results.append(1)
            continue

        visited = {u}
        frontier = {u}
        found = False
        for dist in range(2, max_dist + 1):
            next_f = set()
            for node in frontier:
                for nb in neighbors[node]:
                    if nb == v:
                        found = True
                        break
                    if nb not in visited:
                        visited.add(nb)
                        next_f.add(nb)
                if found:
                    break
            if found:
                results.append(dist)
                break
            frontier = next_f
            if not frontier:
                break

        if not found:
            results.append(max_dist + 1)
    return results


def _structural_features_cpu(
    edge_pairs_cpu, neighbors, deg_cpu, log_deg_cpu, max_dist=5
):
    src_list = edge_pairs_cpu[:, 0].tolist()
    dst_list = edge_pairs_cpu[:, 1].tolist()

    spd_vals = _compute_spd_bfs(neighbors, src_list, dst_list, max_dist)

    cn_vals, aa_vals, jac_vals = [], [], []
    for u, v in zip(src_list, dst_list):
        nu = neighbors[u]
        nv = neighbors[v]
        common = nu & nv
        cn = float(len(common))
        aa = sum(1.0 / log_deg_cpu[w].item() for w in common) if common else 0.0
        union = len(nu | nv)
        jac = cn / union if union > 0 else 0.0
        cn_vals.append(cn)
        aa_vals.append(aa)
        jac_vals.append(jac)

    return (
        torch.tensor(spd_vals, dtype=torch.float32),
        torch.tensor(cn_vals, dtype=torch.float32),
        torch.tensor(aa_vals, dtype=torch.float32),
        torch.tensor(jac_vals, dtype=torch.float32),
    )


# SIEG Model
# https://github.com/anonymous20221001/SIEG_OGB/blob/master/OGB_VESSEL_SIEG.pdf
class SIEGLinkPredictor(nn.Module):
    """
    SIEG double-tower link predictor - tuned for Dataset C.

    Tower 1 (global) : 3-layer GCN + LayerNorm + skip connection
    Tower 2 (local)  : Transformer attention between the two centered
                       nodes only, with structural bias
                       S_ij = b_SPD + b_CN + f_AA + f_Jac
    Decoder          : MLP( h_src || h_dst || attn_value )
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 256,
        out: int = 128,
        dropout: float = 0.5,
        max_spd: int = 5,
    ):
        super().__init__()
        self.dropout = dropout
        self.max_spd = max_spd
        self.out = out

        # Tower 1: GCN + LayerNorm
        self.gcn1 = GCNConv(in_channels, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.gcn3 = GCNConv(hidden, out)
        self.norm3 = nn.LayerNorm(out)
        self.skip = nn.Linear(in_channels, out, bias=False)

        # Tower 2: Q/K/V projections
        self.W_Q = nn.Linear(out, out, bias=False)
        self.W_K = nn.Linear(out, out, bias=False)
        self.W_V = nn.Linear(out, out, bias=False)

        # Structural bias terms
        self.spd_embed = nn.Embedding(max_spd + 2, 1)
        self.cn_mlp = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
        self.aa_mlp = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
        self.jac_mlp = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))

        # Decoder MLP
        self.mlp = nn.Sequential(
            nn.Linear(out * 3, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

        # Neighbor-set cache
        self._neighbors = None
        self._deg_cpu = None
        self._log_deg_cpu = None
        self._cached_ei = None
        self._cached_N = None

    def _refresh_cache(self, edge_index, num_nodes):
        ei_cpu = edge_index.cpu()
        if (
            self._cached_ei is not None
            and self._cached_N == num_nodes
            and torch.equal(self._cached_ei, ei_cpu)
        ):
            return
        self._neighbors = _build_neighbor_sets(ei_cpu, num_nodes)
        deg = torch.zeros(num_nodes)
        for u, nbs in enumerate(self._neighbors):
            deg[u] = len(nbs)
        deg = deg.clamp(min=1.0)
        self._deg_cpu = deg
        self._log_deg_cpu = deg.log().clamp(min=1e-8)
        self._cached_ei = ei_cpu
        self._cached_N = num_nodes

    def encode(self, x, edge_index):
        x_in = x
        h = self.norm1(F.relu(self.gcn1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.norm2(F.relu(self.gcn2(h, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.norm3(self.gcn3(h, edge_index) + self.skip(x_in))
        return h  # [N, out]

    def _pairwise_attention(self, h, edge_pairs):
        device = h.device
        src, dst = edge_pairs[:, 0], edge_pairs[:, 1]
        h_s, h_t = h[src], h[dst]

        Q = self.W_Q(h_s)
        K = self.W_K(h_t)
        V = self.W_V(h_t)
        A = (Q * K).sum(dim=-1, keepdim=True) / (self.out**0.5)  # [E, 1]

        spd_t, cn_t, aa_t, jac_t = _structural_features_cpu(
            edge_pairs.cpu(),
            self._neighbors,
            self._deg_cpu,
            self._log_deg_cpu,
            self.max_spd,
        )

        b_spd = self.spd_embed(spd_t.long().clamp(0, self.max_spd + 1).to(device))
        b_cn = self.cn_mlp(cn_t.unsqueeze(1).to(device))
        f_aa = self.aa_mlp(aa_t.unsqueeze(1).to(device))
        f_jac = self.jac_mlp(jac_t.unsqueeze(1).to(device))

        S_ij = b_spd + b_cn + f_aa + f_jac
        A_weight = torch.sigmoid(A + S_ij)
        return A_weight * V  # [E, out]

    def decode(self, h, edge_pairs):
        src, dst = edge_pairs[:, 0], edge_pairs[:, 1]
        attn = self._pairwise_attention(h, edge_pairs)
        return self.mlp(torch.cat([h[src], h[dst], attn], dim=-1)).squeeze(-1)

    def forward(self, x, edge_index, edge_pairs):
        self._refresh_cache(edge_index, x.size(0))
        return self.decode(self.encode(x, edge_index), edge_pairs)


# Loss helpers
def bce_loss(pos_scores, neg_scores, device):
    """Standard BCE with logits — what the SIEG paper actually uses."""
    labels = torch.cat(
        [
            torch.ones(pos_scores.shape[0], device=device),
            torch.zeros(neg_scores.shape[0], device=device),
        ]
    )
    return F.binary_cross_entropy_with_logits(
        torch.cat([pos_scores, neg_scores]), labels
    )


def margin_ranking_loss(pos_scores, neg_scores, margin=1.0):
    """
    Max-margin ranking: loss = mean( max(0, margin - pos + neg) )
    pos_scores : [P]
    neg_scores : [P]  (1 neg per pos; we subsample to min length)
    """
    return F.relu(margin - pos_scores + neg_scores).mean()
