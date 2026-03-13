import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np

from config import DefaultConfig

configs = DefaultConfig()


class BiLSTMAttentionLayer(nn.Module):
    def __init__(self, num_hidden, num_layers):
        super(BiLSTMAttentionLayer, self).__init__()
        self.feature_dim = configs.feature_dim

        self.encoder = nn.LSTM(input_size=self.feature_dim,
                               hidden_size=num_hidden,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True)

        self.w_omega = nn.Parameter(torch.Tensor(num_hidden * 2, num_hidden * 2))
        self.u_omega = nn.Parameter(torch.Tensor(num_hidden * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, inputs):
        outputs, _ = self.encoder(inputs)
        # print(outputs.shape)
        u = torch.tanh(torch.matmul(outputs, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        scored_x = outputs * att_score
        outs = torch.sum(scored_x, dim=1)
        # print(outs.shape)
        return outs
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedBiLSTMAttentionLayer(nn.Module):
    """
    Enhanced global feature extractor:
      - BiLSTM encoder (same as before)
      - optional Transformer encoder (multi-head self-attention) to capture long-range interactions
      - LayerNorm + residual + dropout
      - Hybrid pooling: attention pooling + mean+max pooling blended with a learnable gate
    Returns:
      tensor of shape [B, 2*num_hidden] (same as original interface).
    """
    def __init__(self, num_hidden, num_layers, use_transformer=True, trans_dropout=0.1, att_dropout=0.1):
        super(ImprovedBiLSTMAttentionLayer, self).__init__()
        self.feature_dim = configs.feature_dim
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.d_model = num_hidden * 2  # bidirectional output dim

        # BiLSTM encoder (same role as original)
        self.encoder = nn.LSTM(input_size=self.feature_dim,
                               hidden_size=num_hidden,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True)

        # small linear to optionally project LSTM outputs before Transformer (identity if not needed)
        self.project_in = nn.Linear(self.d_model, self.d_model)

        # choose head count that divides d_model (prefer 8,4,2,1)
        def _choose_nhead(d_model):
            for h in (8, 4, 2, 1):
                if d_model % h == 0:
                    return h
            return 1
        nhead = _choose_nhead(self.d_model)

        # optional Transformer encoder layer to enhance long-range interactions
        self.use_transformer = use_transformer
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                       nhead=nhead,
                                                       dim_feedforward=self.d_model * 4,
                                                       dropout=trans_dropout,
                                                       activation='relu',
                                                       batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)  # you can stack more if desired

        # LayerNorms for stability
        self.ln_lstm = nn.LayerNorm(self.d_model)
        if use_transformer:
            self.ln_trans = nn.LayerNorm(self.d_model)

        # Attention pooling params (learnable)
        self.w_omega = nn.Linear(self.d_model, self.d_model)
        self.u_omega = nn.Linear(self.d_model, 1)

        # Gate to blend attention-pooled vector with mean+max pooling
        self.gate_proj = nn.Linear(self.d_model * 3, 1)  # inputs: [att_pool, mean_pool, max_pool]

        # final fusion
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        # dropout
        self.dropout = nn.Dropout(att_dropout)

        # init small params similarly to original (optional)
        nn.init.uniform_(self.w_omega.weight, -0.1, 0.1)
        nn.init.uniform_(self.u_omega.weight, -0.1, 0.1)
        if self.w_omega.bias is not None:
            nn.init.constant_(self.w_omega.bias, 0)
        if self.u_omega.bias is not None:
            nn.init.constant_(self.u_omega.bias, 0)

    def forward(self, inputs):
        """
        inputs: (B, L, feature_dim)
        returns: (B, d_model) where d_model == 2 * num_hidden
        """
        # BiLSTM
        outputs, _ = self.encoder(inputs)  # (B, L, 2*num_hidden)
        outputs = self.ln_lstm(outputs)  # LayerNorm over features

        # optional Transformer self-attention (residual)
        if self.use_transformer:
            proj = self.project_in(outputs)
            trans_out = self.transformer(proj)  # (B, L, d_model)
            trans_out = self.ln_trans(trans_out)
            outputs = outputs + self.dropout(trans_out)  # residual + dropout

        # Attention pooling (learned)
        u = torch.tanh(self.w_omega(outputs))       # (B, L, d_model)
        att_logits = self.u_omega(u).squeeze(-1)    # (B, L)
        att_score = F.softmax(att_logits, dim=1).unsqueeze(-1)  # (B, L, 1)
        att_pool = torch.sum(outputs * att_score, dim=1)       # (B, d_model)

        # mean + max pooling
        mean_pool = torch.mean(outputs, dim=1)    # (B, d_model)
        max_pool, _ = torch.max(outputs, dim=1)   # (B, d_model)

        # gate-based blending of pools
        combined_cat = torch.cat([att_pool, mean_pool, max_pool], dim=1)  # (B, 3*d_model)
        gate = torch.sigmoid(self.gate_proj(combined_cat))  # (B, 1)
        fused = gate * att_pool + (1.0 - gate) * (0.5 * (mean_pool + max_pool))  # (B, d_model)

        # final projection + residual from mean to stabilize
        out = self.out_proj(fused) + mean_pool
        out = F.relu(out)
        out = self.dropout(out)

        # ensure shape (B, d_model)
        return out


class ResCNN(nn.Module):
    def __init__(self, in_planes, planes, window, stride=1):
        super(ResCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=2 * window + 1, stride=stride, padding=window, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=2 * window + 1, stride=stride, padding=window, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=2 * window + 1, stride=stride, padding=window, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv1(out))
        out += x
        out = F.relu(out)
        return out


class NodeAverageLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate):
        super(NodeAverageLayer, self).__init__()
        self.center_weight = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.nh_weight = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = Parameter(torch.FloatTensor(out_dim))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.reset_parameters()

    def reset_parameters(self):
        center_std = 1. / np.prod(self.center_weight.shape[0:-1])
        nh_std = 1. / np.prod(self.nh_weight.shape[0:-1])
        self.center_weight.data.uniform_(-center_std, center_std)
        self.nh_weight.data.uniform_(-nh_std, nh_std)
        self.bias.data.fill_(0)

    def forward(self, vertex, nh_indices):
        nh_size = nh_indices.shape[1]
        zc = torch.mm(vertex, self.center_weight)
        zn = torch.mm(vertex, self.nh_weight)
        zn = zn[torch.squeeze(nh_indices)]
        zn = torch.sum(zn, axis=1)
        zn = torch.div(zn, nh_size)
        z = zc + zn + self.bias
        z = self.activation(z)
        z = self.dropout(z)
        return z

class NodeEdgeEnhancedLayer(nn.Module):
    """
    center + mean(neighbors) + mean(edge) base, with:
      - linear bottleneck for each term
      - residual connection
      - layernorm + activation + dropout
    Assumes:
      vertex: [N, in_dim]
      edge:   [N, nh_size, 2]
      nh_indices: [N, nh_size] (LongTensor)
    """
    def __init__(self, in_dim, out_dim, dropout_rate=0.2, use_residual=True):
        super().__init__()
        self.use_residual = use_residual and (in_dim == out_dim)
        self.center_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.nh_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.edge_lin = nn.Linear(2, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # init
        nn.init.xavier_uniform_(self.center_lin.weight)
        nn.init.xavier_uniform_(self.nh_lin.weight)
        nn.init.xavier_uniform_(self.edge_lin.weight)

    def forward(self, vertex, edge, nh_indices):
        """
        Robust forward:
        - accepts nh_indices shaped [N, nh] or [N, nh, 1] or [N] (will unsqueeze)
        - supports nh_indices padded with -1 (meaning 'no neighbor')
        - ensures index dtype/device correctness
        - handles edge.size(1) != nh_size by slicing/padding
        vertex: [N, in_dim]
        edge:   [N, edge_nh_size, 2]
        nh_indices: [N, nh_size] or [N, nh_size, 1]
        """
        # --- normalize nh_indices to shape [N, nh_size] and dtype long on same device ---
        if nh_indices is None:
            raise ValueError("nh_indices is None")
        # move to same device as vertex if needed
        device = vertex.device
        if not nh_indices.device == device:
            nh_indices = nh_indices.to(device)
        # squeeze trailing singleton dim if present
        if nh_indices.dim() == 3 and nh_indices.size(-1) == 1:
            nh_indices = nh_indices.squeeze(-1)
        # if 1D, make it [N, 1]
        if nh_indices.dim() == 1:
            nh_indices = nh_indices.unsqueeze(1)
        if nh_indices.dim() != 2:
            raise ValueError(f"nh_indices must be 2D after normalization, got shape {nh_indices.shape}")

        nh_indices = nh_indices.long()

        N, nh_size = nh_indices.shape  # now safe

        # --- compute transformed features ---
        zc = self.center_lin(vertex)   # [N, out_dim]
        zn = self.nh_lin(vertex)       # [N, out_dim]

        # make sure edge has compatible neighbor dim
        edge_nh = edge.size(1)
        if edge_nh != nh_size:
            # if edge has more neighbors, slice; if fewer, pad with zeros
            if edge_nh > nh_size:
                edge = edge[:, :nh_size, :].contiguous()
            else:
                pad_size = nh_size - edge_nh
                pad = edge.new_zeros((N, pad_size, edge.size(2)))
                edge = torch.cat([edge, pad], dim=1)

        # --- gather neighbor features safely, handle -1 padding ---
        # treat -1 in nh_indices as padding: mask them out
        mask = (nh_indices != -1)                 # [N, nh_size] bool
        idx = nh_indices.clone()
        idx[~mask] = 0                            # avoid OOB when indexing

        # advanced indexing: zn[idx] -> [N, nh_size, out_dim]
        zn_nb = zn[idx]                           # [N, nh_size, out_dim]
        # zero out padded neighbor slots
        mask_f = mask.unsqueeze(-1).type_as(zn_nb)  # [N, nh_size, 1]
        zn_nb = zn_nb * mask_f

        # mean over valid neighbors (avoid divide by 0)
        valid_counts = mask.sum(dim=1).clamp_min(1).unsqueeze(1).type_as(zn_nb)  # [N,1]
        zn_mean = zn_nb.sum(dim=1) / valid_counts   # [N, out_dim]

        # edge features
        ze = self.edge_lin(edge)                   # [N, nh_size, out_dim]
        ze = ze * mask_f                           # zero out padded positions
        ze_mean = ze.sum(dim=1) / valid_counts     # [N, out_dim]

        # --- combine ---
        z = zc + zn_mean + ze_mean + self.bias
        z = self.norm(z)
        z = self.act(z)
        z = self.dropout(z)

        if self.use_residual:
            # residual only when dims match
            if vertex.shape[1] == z.shape[1]:
                z = z + vertex
            # else: skip residual (you already set use_residual=False when dims differ)

        return z

class NodeEdgeAttentionLayer(nn.Module):
    """
    Edge-aware additive attention aggregation.
    z_center = Wc x_center
    For each neighbor j: score = a([Wq x_center || Wk x_neighbor || We edge_ij])
    aggregate = sum_j softmax(score_j) * V x_neighbor_or_edge
    """
    def __init__(self, in_dim, out_dim, dropout_rate=0.2, heads=1):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.center_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.nei_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.edge_lin = nn.Linear(2, out_dim, bias=False)
        # attention scorer
        self.attn = nn.Linear(out_dim * 3, 1)  # concat(center, neighbor, edge) -> score
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(out_dim)

        nn.init.xavier_uniform_(self.attn.weight)

    def forward(self, vertex, edge, nh_indices, return_attn=False):
        """
        Robust forward that tolerates mismatched nh_size between `nh_indices` and `edge`.
        Assumptions:
        - vertex: [N, in_dim]
        - edge:   [N, edge_nh_size, edge_feat_dim]   (edge_feat_dim == 2 in your case)
        - nh_indices: [N, nh_indices_size] (LongTensor), may contain -1 as padding
        Behavior:
        - we take edge_nh_size = edge.size(1) as the canonical neighbors count
        - if nh_indices has more cols -> we slice nh_indices[:, :edge_nh_size]
        - if nh_indices has fewer cols -> we pad nh_indices with self-index and mask accordingly
        - supports nh_indices padded with -1 (means 'no neighbor')
        """
        N = vertex.size(0)
        edge_nh = edge.size(1)           # canonical neighbor count from edge
        idx_nh = nh_indices.squeeze(-1) if nh_indices.dim() == 3 else nh_indices  # be safe

        # ensure idx_nh is [N, K]
        if idx_nh.size(1) > edge_nh:
            # slice extra indices (they would not have edge data)
            idx_nh = idx_nh[:, :edge_nh]
        elif idx_nh.size(1) < edge_nh:
            # pad indices with self-index (or -1) to match edge length
            pad_size = edge_nh - idx_nh.size(1)
            # use -1 as padding sentinel (we'll mask them out)
            pad = torch.full((N, pad_size), -1, dtype=idx_nh.dtype, device=idx_nh.device)
            idx_nh = torch.cat([idx_nh, pad], dim=1)  # [N, edge_nh]

        nh_size = edge_nh  # now matches edge

        # Compute transformed features
        zc = self.center_lin(vertex)          # [N, out_dim]
        zn = self.nei_lin(vertex)             # [N, out_dim]
        # handle -1 padding: replace -1 by 0 temporarily for indexing (we'll mask later)
        idx_for_gather = idx_nh.clone()
        mask = (idx_for_gather != -1)         # [N, nh_size] bool
        idx_for_gather[~mask] = 0             # safe index to avoid indexing error

        # gather neighbor features => [N, nh_size, out_dim]
        zn_nb = zn[idx_for_gather]            # advanced indexing

        # edge -> [N, nh_size, edge_feat]  (must match nh_size)
        ze = self.edge_lin(edge)              # [N, nh_size, out_dim]

        # mask out padded neighbors: set their features to 0 so they don't affect sums
        mask_f = mask.unsqueeze(-1).type_as(zn_nb)  # [N, nh_size, 1]
        zn_nb = zn_nb * mask_f
        ze = ze * mask_f

        # compute attention scores (additive)
        zc_exp = zc.unsqueeze(1).expand(-1, nh_size, -1)  # [N, nh_size, out_dim]
        attn_in = torch.cat([zc_exp, zn_nb, ze], dim=-1)  # [N, nh_size, 3*out_dim]
        scores = self.attn(attn_in).squeeze(-1)           # [N, nh_size]

        # mask scores so padded slots get -inf before softmax
        scores = scores.masked_fill(~mask, float("-1e9"))
        alpha = F.softmax(scores, dim=1)                  # [N, nh_size]

        # weighted neighbor message (use neighbor features + edge)
        nei_message = (zn_nb + ze)                        # padded slots are zeroed
        agg = (alpha.unsqueeze(-1) * nei_message).sum(dim=1)  # [N, out_dim]

        out = zc + self.out_proj(agg)
        out = self.norm(self.act(out))
        out = self.dropout(out)

        if return_attn:
            return out, alpha
        return out


class NodeEdgeAverageLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate):
        super(NodeEdgeAverageLayer, self).__init__()
        self.center_weight = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.nh_weight = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.edge_weight = Parameter(torch.FloatTensor(2, out_dim))
        self.bias = Parameter(torch.FloatTensor(out_dim))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.reset_parameters()

    def reset_parameters(self):
        center_std = 1. / np.prod(self.center_weight.shape[0:-1])
        nh_std = 1. / np.prod(self.nh_weight.shape[0:-1])
        edge_std = 1. / np.prod(self.edge_weight.shape[0:-1])
        self.center_weight.data.uniform_(-center_std, center_std)
        self.nh_weight.data.uniform_(-nh_std, nh_std)
        self.edge_weight.data.uniform_(-edge_std, edge_std)
        self.bias.data.fill_(0)

    def forward(self, vertex, edge, nh_indices):
        nh_size = nh_indices.shape[1]
        zc = torch.mm(vertex, self.center_weight)
        zn = torch.mm(vertex, self.nh_weight)
        ze = torch.tensordot(edge, self.edge_weight, ([-1], [0]))

        zn = zn[torch.squeeze(nh_indices)]
        zn = torch.div(torch.sum(zn, 1), nh_size)
        ze = torch.div(torch.sum(ze, 1), nh_size)

        z = zc + zn + ze + self.bias
        z = self.activation(z)
        z = self.dropout(z)
        return z