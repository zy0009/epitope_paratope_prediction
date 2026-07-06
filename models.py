from layers import *
from config import DefaultConfig
import math
configs = DefaultConfig()


class ResCNNModel(nn.Module):
    def __init__(self):
        super(ResCNNModel, self).__init__()
        window = configs.window_size
        feature_dim = configs.feature_dim
        mlp_dim = configs.mlp_dim
        dropout_rate = configs.dropout_rate
        self.rescnn = ResCNN(1, 1, window)
        self.linear1 = nn.Sequential(
            nn.Linear(feature_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(mlp_dim, 1),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, vertex, indices):
        out = torch.unsqueeze(torch.unsqueeze(vertex, 0), 0)
        out = self.rescnn(out)
        out = torch.squeeze(out)
        out = out[indices]
        out = self.linear1(out)
        out = self.linear2(out)
        return out


class BiLSTMResCNN(nn.Module):
    def __init__(self):
        super(BiLSTMResCNN, self).__init__()
        num_hidden = configs.num_hidden
        num_layer = configs.num_layer
        dropout_rate = configs.dropout_rate
        feature_dim = configs.feature_dim
        window = configs.window_size

        self.bilstm = BiLSTMAttentionLayer(num_hidden, num_layer)
        self.rescnn = ResCNN(1, 1, window)

        self.linear1 = nn.Sequential(
            nn.Linear(feature_dim + 2 * num_hidden, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, vertex, indices):
        batch_size = len(indices)
        global_vertex = vertex.repeat(batch_size, 1, 1)
        global_out = self.bilstm(global_vertex)

        local_out = torch.unsqueeze(torch.unsqueeze(vertex, 0), 0)
        local_out = self.rescnn(local_out)
        local_out = torch.squeeze(local_out)
        local_out = local_out[indices]

        out = torch.cat((local_out, global_out), 1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out


class NodeAverageModel(nn.Module):
    def __init__(self):
        super(NodeAverageModel, self).__init__()
        dropout_rate = configs.dropout_rate
        feature_dim = configs.feature_dim
        mlp_dim = configs.mlp_dim
        self.hidden_dims = configs.hidden_dim.copy()
        self.hidden_dims.insert(0, feature_dim)
        self.node_avers = nn.ModuleList([NodeAverageLayer(self.hidden_dims[i], self.hidden_dims[i + 1], dropout_rate)
                                         for i in range(len(self.hidden_dims[:-1]))])
        self.linear1 = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(mlp_dim, 1),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, vertex, nh_indices, indices):
        for i in range(len(self.hidden_dims[:-1])):
            vertex = self.node_avers[i](vertex, nh_indices)
        out = vertex[indices]
        out = self.linear1(out)
        out = self.linear2(out)
        return out


class BiLSTMNodeAverageModel(nn.Module):
    def __init__(self):
        super(BiLSTMNodeAverageModel, self).__init__()
        num_hidden = configs.num_hidden
        num_layer = configs.num_layer
        dropout_rate = configs.dropout_rate
        feature_dim = configs.feature_dim
        mlp_dim = configs.mlp_dim
        self.hidden_dims = configs.hidden_dim.copy()
        self.hidden_dims.insert(0, feature_dim)
        self.bilstm = BiLSTMAttentionLayer(num_hidden, num_layer)
        self.node_avers = nn.ModuleList([NodeEdgeAttentionLayer(self.hidden_dims[i], self.hidden_dims[i + 1], dropout_rate)
                                         for i in range(len(self.hidden_dims[:-1]))])
        self.linear1 = nn.Sequential(
            nn.Linear(self.hidden_dims[-1] + 2 * num_hidden, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(mlp_dim, 1),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, vertex, nh_indices, indices):
        batch_size = len(indices)
        global_vertex = vertex.repeat(batch_size, 1, 1)
        global_out = self.bilstm(global_vertex)
        for i in range(len(self.hidden_dims[:-1])):
            vertex = self.node_avers[i](vertex, nh_indices)
        local_out = vertex[indices]
        out = torch.cat((global_out, local_out), 1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out


class NodeEdgeAverageModel(nn.Module):
    def __init__(self):
        super(NodeEdgeAverageModel, self).__init__()
        dropout_rate = configs.dropout_rate
        feature_dim = configs.feature_dim
        mlp_dim = configs.mlp_dim
        self.hidden_dims = configs.hidden_dim.copy()
        self.hidden_dims.insert(0, feature_dim)
        self.nodeedge_avers = nn.ModuleList([NodeEdgeAverageLayer(self.hidden_dims[i], self.hidden_dims[i + 1], dropout_rate)
                                         for i in range(len(self.hidden_dims[:-1]))])
        self.linear1 = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(mlp_dim, 1),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, vertex, edge, nh_indices, indices):
        for i in range(len(self.hidden_dims[:-1])):
            vertex = self.nodeedge_avers[i](vertex, edge, nh_indices)
        out = vertex[indices]
        out = self.linear1(out)
        out = self.linear2(out)
        return out

class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer attention over *neighborhoods*.
    vertex: (N, in_dim)          -- node features
    nh_indices: (N, k)           -- neighbor indices for each node; use -1 for padding
    returns: (N, out_dim)        -- updated node features
    """
    def __init__(self, in_dim, out_dim, n_heads=8, dropout=0.1, ff_hidden_mult=4, use_layer_norm=True):
        super(GraphTransformerLayer, self).__init__()
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm

        # Q comes from center node; K,V come from neighbors (including possibly the center itself if nh_indices includes it)
        self.q_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.k_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.v_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.out_lin = nn.Linear(out_dim, out_dim)

        # optional residual projection if dims differ
        if in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)
        else:
            self.res_proj = None

        # LayerNorm(s) and FFN
        if use_layer_norm:
            self.attn_norm = nn.LayerNorm(out_dim)
            self.ffn_norm = nn.LayerNorm(out_dim)
        else:
            self.attn_norm = None
            self.ffn_norm = None

        ff_hidden = out_dim * ff_hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, ff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, out_dim),
            nn.Dropout(dropout)
        )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, vertex, edge, nh_indices):
        """
        vertex: tensor (N, in_dim)
        edge: unused (kept for API compatibility)
        nh_indices: LongTensor (N, k) neighbor indices; use -1 for padding
        """
        N = vertex.size(0)
        device = vertex.device
        k = nh_indices.size(1)

        # make sure nh_indices is a LongTensor on same device
        # If nh_indices is numpy array, convert before calling model: torch.from_numpy(...).long()
        nh = nh_indices.long().to(device)  # do not clone yet, keep original for mask check

        # mask padded positions (assume padding == -1)
        mask = (nh < 0)  # (N, k) boolean

        # replace padded indices with a valid index (0) to avoid index error during gather;
        # their effect will be zeroed-out by mask later when computing attention.
        nh_clamped = nh.clone()
        if mask.any():
            nh_clamped[mask] = 0

        # robust neighbor gather: index_select + reshape (avoids non-contiguous .view issues)
        flat_idx = nh_clamped.reshape(-1)               # (N*k,)
        neighbors = vertex.index_select(0, flat_idx)    # (N*k, in_dim)
        neighbors = neighbors.reshape(N, k, -1)         # (N, k, in_dim)

        # Linear projections
        q = self.q_lin(vertex).view(N, self.n_heads, self.head_dim)        # (N, H, D)
        k_tensor = self.k_lin(neighbors).view(N, k, self.n_heads, self.head_dim)  # (N, k, H, D)
        v_tensor = self.v_lin(neighbors).view(N, k, self.n_heads, self.head_dim)  # (N, k, H, D)

        # Compute attention scores
        q_exp = q.unsqueeze(1)  # (N,1,H,D)
        scores = (q_exp * k_tensor).sum(-1) / math.sqrt(self.head_dim)  # (N, k, H)

        # Mask out padded neighbors
        if mask.any():
            scores = scores.masked_fill(mask.unsqueeze(-1), float('-inf'))

        attn = F.softmax(scores, dim=1)  # (N, k, H)
        attn = self.attn_dropout(attn)
        attn_exp = attn.unsqueeze(-1)  # (N, k, H, 1)

        weighted = (attn_exp * v_tensor).sum(1)  # (N, H, D)
        weighted = weighted.reshape(N, self.out_dim)
        out = self.out_lin(weighted)

        # Residual + Norm
        res = vertex if self.res_proj is None else self.res_proj(vertex)
        out = out + res
        if self.attn_norm is not None:
            out = self.attn_norm(out)

        # FFN block
        ffn_out = self.ffn(out)
        out = out + ffn_out
        if self.ffn_norm is not None:
            out = self.ffn_norm(out)

        return out

class BiLSTMGraphTransformerModel(nn.Module):
    def __init__(self):
        super(BiLSTMGraphTransformerModel, self).__init__()
        num_hidden = configs.num_hidden
        num_layer = configs.num_layer
        dropout_rate = configs.dropout_rate
        feature_dim = configs.feature_dim
        self.hidden_dims = configs.hidden_dim.copy()
        self.hidden_dims.insert(0, feature_dim)
        self.bilstm = BiLSTMAttentionLayer(num_hidden, num_layer)
        self.node_edge_avers = nn.ModuleList([
            GraphTransformerLayer(self.hidden_dims[i], self.hidden_dims[i + 1],
                                  n_heads=8, dropout=dropout_rate)
            for i in range(len(self.hidden_dims[:-1]))
        ])
        self.linear1 = nn.Sequential(
            nn.Linear(self.hidden_dims[-1] + 2 * num_hidden, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(512, 1),
            nn.Dropout(dropout_rate)
        )

        self.cross_attention = GlobalLocalCrossAttention(
            global_dim=2 * num_hidden,
            local_dim=self.hidden_dims[-1],
            dropout=dropout_rate,
        )

    def forward(self, vertex, edge, nh_indices, indices):
        batch_size = len(indices)
        # print(batch_size)
        global_vertex = vertex.repeat(batch_size, 1, 1)
        # print(global_vertex)
        global_out = self.bilstm(global_vertex)
        # print(global_out)
        for i in range(len(self.hidden_dims[:-1])):
            vertex = self.node_edge_avers[i](vertex, edge, nh_indices)
            # print(vertex)
        local_out = vertex[indices]
        # print(local_out)
        global_out, local_out = self.cross_attention(global_out, local_out)

        combined = torch.cat([global_out, local_out], dim=1)
        out = self.linear1(combined)
        out = self.linear2(out)

        return out, global_out, local_out


class GlobalLocalCrossAttention(nn.Module):
    def __init__(self, global_dim, local_dim, attn_dim=128, num_heads=4, dropout=0.1):
        super(GlobalLocalCrossAttention, self).__init__()
        if attn_dim % num_heads != 0:
            raise ValueError("attn_dim must be divisible by num_heads")
        self.global_q = nn.Linear(global_dim, attn_dim)
        self.global_kv = nn.Linear(global_dim, attn_dim)
        self.local_q = nn.Linear(local_dim, attn_dim)
        self.local_kv = nn.Linear(local_dim, attn_dim)
        self.global_to_local_attn = nn.MultiheadAttention(
            attn_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.local_to_global_attn = nn.MultiheadAttention(
            attn_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.global_out = nn.Linear(attn_dim, global_dim)
        self.local_out = nn.Linear(attn_dim, local_dim)
        self.global_norm = nn.LayerNorm(global_dim)
        self.local_norm = nn.LayerNorm(local_dim)
        self.dropout = nn.Dropout(dropout)

    def _augmented_neighbors(self, indices, nh_indices, num_nodes):
        idx_nh = nh_indices.squeeze(-1) if nh_indices.dim() == 3 else nh_indices
        target_nh = idx_nh[indices]
        self_idx = indices.unsqueeze(1)
        neigh = torch.cat([self_idx, target_nh], dim=1)
        mask = (neigh >= 0) & (neigh < num_nodes)
        safe_neigh = neigh.clamp(min=0, max=max(num_nodes - 1, 0))
        return safe_neigh, mask

    def forward(self, global_feat, local_feat, indices=None, nh_indices=None):
        if indices is None or nh_indices is None:
            global_q = self.global_q(global_feat).unsqueeze(1)
            global_kv = self.global_kv(global_feat).unsqueeze(1)
            local_q = self.local_q(local_feat).unsqueeze(1)
            local_kv = self.local_kv(local_feat).unsqueeze(1)
        else:
            neigh, mask = self._augmented_neighbors(indices, nh_indices, local_feat.size(0))
            target_global = global_feat[indices]
            target_local = local_feat[indices]
            global_q = self.global_q(target_global).unsqueeze(1)
            local_q = self.local_q(target_local).unsqueeze(1)
            global_kv = self.global_kv(global_feat)[neigh]
            local_kv = self.local_kv(local_feat)[neigh]

        key_padding_mask = None if indices is None or nh_indices is None else ~mask
        global_ctx, _ = self.global_to_local_attn(
            global_q, local_kv, local_kv, key_padding_mask=key_padding_mask
        )
        local_ctx, _ = self.local_to_global_attn(
            local_q, global_kv, global_kv, key_padding_mask=key_padding_mask
        )

        base_global = global_feat if indices is None or nh_indices is None else target_global
        base_local = local_feat if indices is None or nh_indices is None else target_local
        global_refined = self.global_norm(
            base_global + self.dropout(self.global_out(global_ctx.squeeze(1)))
        )
        local_refined = self.local_norm(
            base_local + self.dropout(self.local_out(local_ctx.squeeze(1)))
        )
        return global_refined, local_refined

class ResidualMLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()

        self.fc = nn.Linear(input_dim, hidden_dim)
        self.res = nn.Linear(input_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

        self.out = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        h = self.fc(x)
        h = self.relu(h)
        h = self.dropout(h)

        h = self.norm(h + self.res(x))

        out = self.out(h)

        return out

class BiLSTMNodeEdgeAverageModel(nn.Module):
    def __init__(self):
        super(BiLSTMNodeEdgeAverageModel, self).__init__()
        num_hidden = configs.num_hidden
        dropout_rate = configs.dropout_rate
        feature_dim = configs.feature_dim
        sequence_dim = getattr(configs, 'sequence_encoder_dim', 2 * num_hidden)
        self.use_cross_attention = getattr(configs, 'use_cross_attention', False)
        self.use_contrastive = getattr(configs, 'use_contrastive', False)
        self.hidden_dims = configs.hidden_dim.copy()
        self.hidden_dims.insert(0, feature_dim)
        expected_sequence_dim = 2 * num_hidden
        if sequence_dim != expected_sequence_dim:
            raise ValueError(
                "sequence_encoder_dim must equal 2 * num_hidden when using HSAE; "
                "got {} and {}".format(sequence_dim, expected_sequence_dim)
            )
        self.sequence_encoder = ImprovedBiLSTMAttentionLayer(
            num_hidden=num_hidden,
            num_layers=configs.num_layer,
            use_bilstm=getattr(configs, 'use_bilstm', True),
            use_transformer=configs.use_transformer,
            use_hybrid_pooling=configs.use_hybrid_pooling,
            transformer_layers=configs.transformer_layers,
            trans_dropout=dropout_rate,
            att_dropout=dropout_rate,
        )
        self.node_edge_avers = nn.ModuleList([NodeEdgeAttentionLayer(self.hidden_dims[i], self.hidden_dims[i + 1], dropout_rate)
                                             for i in range(len(self.hidden_dims[:-1]))])
        self.pred_head = ResidualMLPHead(
            input_dim=self.hidden_dims[-1] + sequence_dim,
            hidden_dim=512,
            dropout=dropout_rate
        )

        if self.use_cross_attention:
            self.cross_attention = GlobalLocalCrossAttention(
                global_dim=sequence_dim,
                local_dim=self.hidden_dims[-1],
                dropout=dropout_rate,
            )
        if self.use_contrastive:
            self.global_contrastive_proj = nn.Linear(sequence_dim, configs.contrastive_dim)
            self.local_contrastive_proj = nn.Linear(self.hidden_dims[-1], configs.contrastive_dim)

    def forward(self, vertex, edge, nh_indices, indices):
        global_tokens = self.sequence_encoder(vertex.unsqueeze(0), return_tokens=True).squeeze(0)
        for i in range(len(self.hidden_dims[:-1])):
            vertex = self.node_edge_avers[i](vertex, edge, nh_indices)
            # print(vertex)
        local_tokens = vertex
        # print(local_out)
        if self.use_cross_attention:
            global_out, local_out = self.cross_attention(global_tokens, local_tokens, indices, nh_indices)
        else:
            global_out = global_tokens[indices]
            local_out = local_tokens[indices]

        out = torch.cat((global_out, local_out), 1)
        out = self.pred_head(out)

        return out, global_out, local_out

    def contrastive_embeddings(self, global_out, local_out):
        if not self.use_contrastive:
            raise RuntimeError("contrastive_embeddings is only available when use_contrastive=True")
        return self.global_contrastive_proj(global_out), self.local_contrastive_proj(local_out)
    
        

class BiLSTMResCNNNodeEdgeAverageModel(nn.Module):
    def __init__(self):
        super(BiLSTMResCNNNodeEdgeAverageModel, self).__init__()
        window = configs.window
        num_hidden = configs.num_hidden
        num_layer = configs.num_layer
        dropout_rate = configs.dropout_rate
        feature_dim = configs.feature_dim

        self.cnnres = ResCNN(1, 1, window)
        self.hidden_dims = configs.hidden_dim.copy()
        self.hidden_dims.insert(0, feature_dim)
        self.bilstm = BiLSTMAttentionLayer(num_hidden, num_layer)
        self.node_edge_avers = nn.ModuleList([NodeEdgeAverageLayer(self.hidden_dims[i], self.hidden_dims[i + 1], dropout_rate)
                                             for i in range(len(self.hidden_dims[:-1]))])
        self.linear1 = nn.Sequential(
            nn.Linear(self.hidden_dims[-1] + 2 * num_hidden, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, vertex, edge, nh_indices, indices):
        batch_size = len(indices)
        global_vertex = vertex.repeat(batch_size, 1, 1)
        global_out = self.bilstm(global_vertex)

        vertex = torch.unsqueeze(torch.unsqueeze(vertex, 0), 0)
        vertex = self.cnnres(vertex)
        vertex = torch.squeeze(vertex)
        for i in range(len(self.hidden_dims[:-1])):
            vertex = self.node_edge_avers[i](vertex, edge, nh_indices)
        local_out = vertex[indices]
        out = torch.cat((global_out, local_out), 1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out
