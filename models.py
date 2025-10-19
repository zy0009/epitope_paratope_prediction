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
        self.hidden_dims = configs.hidden_dim
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
        self.hidden_dims = configs.hidden_dim
        self.hidden_dims.insert(0, feature_dim)
        self.bilstm = BiLSTMAttentionLayer(num_hidden, num_layer)
        self.node_avers = nn.ModuleList([NodeAverageLayer(self.hidden_dims[i], self.hidden_dims[i + 1], dropout_rate)
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
        self.hidden_dims = configs.hidden_dim
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
        self.hidden_dims = configs.hidden_dim
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

        # 新增对比学习投影头
        self.global_proj = nn.Linear(2 * num_hidden, 128)
        self.local_proj = nn.Linear(self.hidden_dims[-1], 128)
        self.temperature = 0.1  # 温度系数

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
        out = torch.cat((global_out, local_out), 1)
        out = self.linear1(out)
        out = self.linear2(out)
        # print(out)

        # 全局特征和局部特征
        global_feat = global_out  # [B, 2*num_hidden]
        local_feat = local_out    # [B, hidden_dims[-1]]
        
        # 拼接后输出主预测
        combined = torch.cat([global_feat, local_feat], dim=1)
        out = self.linear1(combined)
        out = self.linear2(out)
        
        # 投影到对比空间
        global_proj = F.normalize(self.global_proj(global_feat), dim=1)
        local_proj = F.normalize(self.local_proj(local_feat), dim=1)
        
        return out, global_proj, local_proj

# class BiLSTMNodeEdgeAverageModel(nn.Module):
#     def __init__(self):
#         super(BiLSTMNodeEdgeAverageModel, self).__init__()
#         num_hidden = configs.num_hidden
#         num_layer = configs.num_layer
#         dropout_rate = configs.dropout_rate
#         feature_dim = configs.feature_dim
#         self.hidden_dims = configs.hidden_dim
#         self.hidden_dims.insert(0, feature_dim)
#         self.bilstm = BiLSTMAttentionLayer(num_hidden, num_layer)
#         self.node_edge_avers = nn.ModuleList([NodeAverageLayer(self.hidden_dims[i], self.hidden_dims[i + 1], dropout_rate)
#                                              for i in range(len(self.hidden_dims[:-1]))])
        
#         # 在这里加 LayerNorm，针对 combined 特征
#         # self.norm_combined = nn.LayerNorm(self.hidden_dims[-1] + 2 * num_hidden)
#         self.linear1 = nn.Sequential(
#             nn.Linear(self.hidden_dims[-1] + 2 * num_hidden, 512),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate)
#         )
#         self.linear2 = nn.Sequential(
#             nn.Linear(512, 1),
#             nn.Dropout(dropout_rate)
#         )

#         # 新增对比学习投影头
#         self.global_proj = nn.Linear(2 * num_hidden, 128)
#         self.local_proj = nn.Linear(self.hidden_dims[-1], 128)
#         self.temperature = 0.1  # 温度系数

#     def forward(self, vertex, edge, nh_indices, indices):
#         """
#         假设：
#         - vertex: Tensor of shape [seq_len, feat_dim] 或 [N, feat_dim]
#         - indices: LongTensor of shape [B]，表示要预测的 residue 的索引（相对于 vertex 的第一维）
#         - NodeAverageLayer 接受 vertex, nh_indices 并返回 vertex'（同第一维长度）
#         - self.bilstm 接受 [B, seq_len, feat_dim] 并返回 [B, 2*num_hidden] 的全局向量（或你实现的池化输出）
#         """

#         # 确定 batch size
#         batch_size = indices.size(0) if isinstance(indices, torch.Tensor) else len(indices)

#         # 如果 vertex 是 [seq_len, feat_dim]，把它扩展成 batch 维
#         if vertex.dim() == 2:
#             # 变为 [1, seq_len, feat_dim] 再 repeat -> [B, seq_len, feat_dim]
#             global_vertex = vertex.unsqueeze(0).repeat(batch_size, 1, 1)
#         elif vertex.dim() == 3:
#             # 已经有 batch 维（请确认第0维是否等于 batch_size）
#             global_vertex = vertex
#             # 如果 global_vertex.shape[0] != batch_size 可能需要调整/报错
#             if global_vertex.size(0) != batch_size:
#                 # 如果你的数据语义不同，可以处理；现在先抛出以便调试
#                 raise ValueError(f"global_vertex batch {global_vertex.size(0)} != indices batch {batch_size}")
#         else:
#             raise ValueError("vertex must be 2D or 3D tensor")

#         # 全局特征，通过 BiLSTM + attention/pooling 得到每个样本的全局表示
#         global_out = self.bilstm(global_vertex)   # 期望 shape: [B, 2*num_hidden]
#         # 检查尺寸
#         # assert global_out.dim() == 2 and global_out.size(0) == batch_size

#         # 层级 NodeAverage 层堆叠（对 vertex 做多层变换）
#         v = vertex
#         for i in range(len(self.hidden_dims) - 1):
#             v = self.node_edge_avers[i](v, nh_indices)   # 返回的 v 形状应为 [seq_len, hidden_dims[i+1]] 或 [N, D]
#         # 从 v 中挑出 local_out
#         local_out = v[indices]  # shape: [B, hidden_dims[-1]]

#         # combined 用于主任务预测
#         combined = torch.cat([global_out, local_out], dim=1)  # shape: [B, 2*num_hidden + hidden_dims[-1]]
#         out = self.linear1(combined)
#         out = self.linear2(out)
#         # out shape: [B, 1] （或 [B,1]）

#         # 对比投影
#         global_proj = F.normalize(self.global_proj(global_out), dim=1)  # [B, 128]
#         local_proj = F.normalize(self.local_proj(local_out), dim=1)     # [B, 128]

#         return out, global_proj, local_proj, combined
class BiLSTMNodeEdgeAverageModel(nn.Module):
    def __init__(self):
        super(BiLSTMNodeEdgeAverageModel, self).__init__()
        num_hidden = configs.num_hidden
        num_layer = configs.num_layer
        dropout_rate = configs.dropout_rate
        feature_dim = configs.feature_dim
        self.hidden_dims = configs.hidden_dim
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
            nn.Dropout(dropout_rate)
        )

        # 新增对比学习投影头
        self.global_proj = nn.Linear(2 * num_hidden, 128)
        self.local_proj = nn.Linear(self.hidden_dims[-1], 128)
        self.temperature = 0.1  # 温度系数

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
        out = torch.cat((global_out, local_out), 1)
        out = self.linear1(out)
        out = self.linear2(out)
        # print(out)

        # 全局特征和局部特征
        global_feat = global_out  # [B, 2*num_hidden]
        local_feat = local_out    # [B, hidden_dims[-1]]
        
        # 拼接后输出主预测
        combined = torch.cat([global_feat, local_feat], dim=1)
        out = self.linear1(combined)
        out = self.linear2(out)
        
        # 投影到对比空间
        global_proj = F.normalize(self.global_proj(global_feat), dim=1)
        local_proj = F.normalize(self.local_proj(local_feat), dim=1)
        
        return out, global_proj, local_proj

class BiLSTMResCNNNodeEdgeAverageModel(nn.Module):
    def __init__(self):
        super(BiLSTMResCNNNodeEdgeAverageModel, self).__init__()
        window = configs.window
        num_hidden = configs.num_hidden
        num_layer = configs.num_layer
        dropout_rate = configs.dropout_rate
        feature_dim = configs.feature_dim

        self.cnnres = ResCNN(1, 1, window)
        self.hidden_dims = configs.hidden_dim
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