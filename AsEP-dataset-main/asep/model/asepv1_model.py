"""
Implement version 1 of the AsEP model
"""

# logging
import logging
# basic
import os
import os.path as osp
import re
import sys
from pathlib import Path
from pprint import pprint
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Set, Tuple, Union)

import numpy as np
import pandas as pd
# torch tools
import torch
import torch.nn as nn
import torch.nn.functional as F
# pyg tools
import torch_geometric as tg
import torch_geometric.transforms as T
import torch_scatter as ts
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch_geometric.data import Batch as PygBatch
from torch_geometric.data import Data as PygData
from torch_geometric.data import Dataset as PygDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader as PygDataLoader
from torch_geometric.nn import GATConv, GCNConv,global_mean_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch, to_undirected

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s {%(pathname)s:%(lineno)d} [%(threadName)s] [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
# custom
from asep.data.asepv1_dataset import AsEPv1Dataset
class _GNNBlock(nn.Module):
    """
    一层 GNN + (可选)LayerNorm + Dropout + (可选)残差
    用来保证放进 ModuleList 里不会报你之前那个 "not a Module subclass" 的错
    """
    def __init__(self, conv: nn.Module, out_c: int,
                 use_ln: bool, dropout_p: float, use_residual: bool):
        super().__init__()
        self.conv = conv
        self.ln = nn.LayerNorm(out_c) if use_ln else nn.Identity()
        self.dp = nn.Dropout(dropout_p)
        self.use_res = bool(use_residual)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, act: nn.Module) -> torch.Tensor:
        pre = x
        h = self.conv(x, edge_index)
        h = self.ln(h)
        h = self.dp(h)
        h = act(h)
        if self.use_res and (pre.shape == h.shape):
            h = h + pre
        return h


class PyGAbAgIntGAE(nn.Module):
    """
    Att-BiLSTM + Cross-Attention GCN 版 Ab–Ag 交互预测模型

    流程：
    1) 输入：batch.x_b, batch.edge_index_b, batch.x_b_batch
              batch.x_g, batch.edge_index_g, batch.x_g_batch
       要求 x_b/x_g 里已经是你的 ESM / ESM-C / one-hot + PSSM 特征
       （也就是说序列信息在特征维度上已经有了）

    2) 序列编码（按图内顺序）：Att-BiLSTM
       - 我们把同一个图里的节点先按出现顺序取出来
       - 送进 BiLSTM
       - 用注意力把序列特征压回来（residue-level）
       - 得到 seq_b / seq_g

    3) 结构编码：GCN / GATv2 / SAGE（可配置）
       - 把 seq_b/seq_g 作为初始节点特征
       - 做若干层图卷积 → 得到 str_b / str_g

    4) 跨模态双向注意力（Ab↔Ag）
       - 让抗体残基看抗原残基
       - 让抗原残基看抗体残基
       - 得到 b_final, g_final

    5) 解码：inner_prod / bilinear / 2-layer fc
       - 按图切片 → (Nb_i, Ng_i)
       - sigmoid → 交给你的 loss

    输出保持和你原来一致
    """
    def __init__(
        self,
        input_ab_dim: int,
        input_ag_dim: int,
        dim_list: List[int],
        act_list: List[str],
        decoder: Optional[Dict[str, Any]] = None,
        try_gpu: bool = True,
        input_ab_act: str = "relu",
        input_ag_act: str = "relu",
        # --- 新增：序列编码器参数 ---
        seq_enc: Optional[Dict[str, Any]] = None,    # {"hidden":256,"layers":1,"dropout":0.1,"use_attn":True}
        # --- 新增：图编码器参数 ---
        encoder: Optional[Dict[str, Any]] = None,     # {"conv":"gcn","dropout":0.2,"use_residual":True,"use_layernorm":True,"heads":2}
        # --- 新增：跨模态 ---
        cross_attention: Optional[Dict[str, Any]] = None,  # {"enable":True,"heads":4,"dropout":0.1}
        # --- 图级对比学习 ---
        graph_cl: Optional[Dict[str, Any]] = None,   # >>> 新增 <<<
        # --- 解码内存控制 ---
        decode_chunk_size: Optional[int] = 32768,
    ):
        super().__init__()
        self.device = torch.device("cuda" if try_gpu and torch.cuda.is_available() else "cpu")

        # 默认配置
        decoder = decoder or {"name": "fc", "hidden": 256, "dropout": 0.2, "bias": True}
        seq_enc = seq_enc or {"hidden": 256, "layers": 1, "dropout": 0.2, "use_attn": True}
        encoder = encoder or {"conv": "gcn", "dropout": 0.2, "use_residual": True, "use_layernorm": True, "heads": 2}
        cross_attention = cross_attention or {"enable": True, "heads": 4, "dropout": 0.2}
        graph_cl = graph_cl or {"enable": True, "proj_dim": 128, "tau": 0.1, "weight": 0.1}  # >>> 新增 <<<

        # 合法性检查
        if not (len(dim_list) == len(act_list) + 1):
            raise ValueError("dim_list length must equal act_list length + 1")
        if decoder["name"] not in ("inner_prod", "fc", "bilinear"):
            raise ValueError("decoder must be one of ['inner_prod','fc','bilinear']")

        self.hparams = {
            "input_ab_dim": input_ab_dim,
            "input_ag_dim": input_ag_dim,
            "dim_list": dim_list,
            "act_list": act_list,
            "decoder": decoder,
            "seq_enc": seq_enc,
            "encoder": encoder,
            "cross_attention": cross_attention,
            "graph_cl": graph_cl,
        }
        self.decode_chunk_size = decode_chunk_size

        # ======================
        # 1. 序列编码器（共享一套结构，抗体和抗原各一份）
        # ======================
        # LSTM 输入维度就是节点特征维度
        self.ab_lstm = nn.LSTM(
            input_size=input_ab_dim,
            hidden_size=seq_enc["hidden"],
            num_layers=seq_enc["layers"],
            batch_first=True,
            bidirectional=True,
            dropout=seq_enc.get("dropout", 0.0) if seq_enc["layers"] > 1 else 0.0,
        )
        self.ag_lstm = nn.LSTM(
            input_size=input_ag_dim,
            hidden_size=seq_enc["hidden"],
            num_layers=seq_enc["layers"],
            batch_first=True,
            bidirectional=True,
            dropout=seq_enc.get("dropout", 0.0) if seq_enc["layers"] > 1 else 0.0,
        )
        self.use_seq_attn = bool(seq_enc.get("use_attn", True))
        if self.use_seq_attn:
            # 用一个简单的 additive attention 做序列内加权
            d_seq = seq_enc["hidden"] * 2
            self.ab_seq_attn = nn.Sequential(
                nn.Linear(d_seq, d_seq),
                nn.Tanh(),
                nn.Linear(d_seq, 1)
            )
            self.ag_seq_attn = nn.Sequential(
                nn.Linear(d_seq, d_seq),
                nn.Tanh(),
                nn.Linear(d_seq, 1)
            )

        # ======================
        # 2. 图编码器（结构通道）
        # ======================
        self.B_encoder_block = self._build_gnn_stack(
            input_dim=input_ab_dim,      # <<< 改这里：不再用 BiLSTM 输出维度
            input_act=input_ab_act,
            dim_list=dim_list,
            act_list=act_list,
            enc_cfg=encoder,
        )
        self.G_encoder_block = self._build_gnn_stack(
            input_dim=input_ag_dim,      # <<< 同上
            input_act=input_ag_act,
            dim_list=dim_list,
            act_list=act_list,
            enc_cfg=encoder,
        )
        d_seq = seq_enc["hidden"] * 2         # BiLSTM 输出维度
        d_gnn = dim_list[-1]                  # 结构编码最终维度

        self.ab_fuse = nn.Sequential(
            nn.Linear(d_seq + d_gnn, d_gnn),
            nn.GELU(),
            nn.LayerNorm(d_gnn),
        )
        self.ag_fuse = nn.Sequential(
            nn.Linear(d_seq + d_gnn, d_gnn),
            nn.GELU(),
            nn.LayerNorm(d_gnn),
        )
        # ======================
        # 3. 跨模态双向注意力（残基级别）
        # ======================
        self.use_xattn = bool(cross_attention.get("enable", True))
        if self.use_xattn:
            d_model = dim_list[-1]
            heads = int(cross_attention.get("heads", 4))
            if d_model % heads != 0:
                raise ValueError(f"d_model({d_model}) must be divisible by heads({heads})")
            self.x_heads = heads
            self.d_head = d_model // heads
            dp = float(cross_attention.get("dropout", 0.1))

            # Ab ← Ag
            self.q_b = nn.Linear(d_model, d_model, bias=False)
            self.k_g = nn.Linear(d_model, d_model, bias=False)
            self.v_g = nn.Linear(d_model, d_model, bias=False)
            self.o_b = nn.Linear(d_model, d_model, bias=False)

            # Ag ← Ab
            self.q_g = nn.Linear(d_model, d_model, bias=False)
            self.k_b = nn.Linear(d_model, d_model, bias=False)
            self.v_b = nn.Linear(d_model, d_model, bias=False)
            self.o_g = nn.Linear(d_model, d_model, bias=False)

            # FFN refine
            self.ffn_b = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2),
                nn.GELU(),
                nn.Dropout(dp),
                nn.Linear(d_model * 2, d_model),
            )
            self.ffn_g = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2),
                nn.GELU(),
                nn.Dropout(dp),
                nn.Linear(d_model * 2, d_model),
            )
            self.x_ln_b = nn.LayerNorm(d_model)
            self.x_ln_g = nn.LayerNorm(d_model)
            self.x_dp = nn.Dropout(dp)

        # ======================
        # 3.5 图级对比学习模块（可选）
        # ======================
        self.use_graph_cl = bool(graph_cl.get("enable", False))    # >>> 新增 <<<
        if self.use_graph_cl:
            d_model = dim_list[-1]
            proj_dim = int(graph_cl.get("proj_dim", 128))
            self.graph_cl_tau = float(graph_cl.get("tau", 0.1))

            # 抗体图表示投影 head
            self.ab_graph_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, proj_dim),
            )
            # 抗原图表示投影 head
            self.ag_graph_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, proj_dim),
            )
        

        # ======================
        # 4. 解码器
        # ======================
        self.decoder = self._init_decoder(decoder)
        self._dc_func = self._make_decoder_fn(decoder)

        # 把所有东西搬到 device
        self.to(self.device)

    # ------------------------------------------------------------------
    # 小工具
    # ------------------------------------------------------------------
    def _act(self, name: Optional[str]) -> nn.Module:
        if name is None:
            return nn.Identity()
        name = name.lower()
        if name == "relu":
            return nn.ReLU(inplace=True)
        if name == "leakyrelu":
            return nn.LeakyReLU(inplace=True)
        if name == "gelu":
            return nn.GELU()
        raise ValueError(f"Unsupported act: {name}")
    

    def _gnn_layer(self, conv_type: str, in_c: int, out_c: int, heads: int = 1):
        conv_type = conv_type.lower()
        if conv_type == "gcn":
            return GCNConv(in_c, out_c, normalize=True)
        if conv_type == "gat":
            if out_c % heads != 0:
                raise ValueError(f"GATv2 out_c({out_c}) must be divisible by heads({heads})")
            return GATConv(in_c, out_c // heads, heads=heads, dropout=0.2)
        if conv_type == "sage":
            return SAGEConv(in_c, out_c)
        raise ValueError(f"Unknown conv type: {conv_type}")

    def _gnn_block(self, in_c: int, out_c: int, enc_cfg: Dict[str, Any]) -> nn.Module:
        conv = self._gnn_layer(enc_cfg["conv"], in_c, out_c, heads=int(enc_cfg.get("heads", 1)))
        use_res = bool(enc_cfg.get("use_residual", True)) and (in_c == out_c)
        use_ln = bool(enc_cfg.get("use_layernorm", True))
        dp = float(enc_cfg.get("dropout", 0.2))
        return _GNNBlock(conv, out_c, use_ln, dp, use_res)

    def _build_gnn_stack(self, input_dim, input_act, dim_list, act_list, enc_cfg):
        blocks = nn.ModuleList()
        acts = nn.ModuleList()
        # 第一层
        blocks.append(self._gnn_block(input_dim, dim_list[0], enc_cfg))
        acts.append(self._act(input_act))
        # 后续层
        for i in range(len(dim_list) - 1):
            blocks.append(self._gnn_block(dim_list[i], dim_list[i + 1], enc_cfg))
            acts.append(self._act(act_list[i]))
        return nn.ModuleDict({
            "blocks": blocks,
            "acts": acts,
        })

    def _run_gnn_stack(self, stack: nn.ModuleDict, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for blk, act in zip(stack["blocks"], stack["acts"]):
            h = blk(h, edge_index, act)
        return h

    # ------------------------------------------------------------------
    # 1) 序列编码：把同一图里的节点拼成序列 -> BiLSTM -> 再摊平回去
    # ------------------------------------------------------------------
    def _run_seq_encoder(self, x: torch.Tensor, batch_idx: torch.Tensor,
                         lstm: nn.LSTM, attn: Optional[nn.Module]) -> torch.Tensor:
        """
        x:     [N, F]  (N = 所有图的节点数之和)
        batch: [N]     (节点属于哪个图)
        返回和 x 一样顺序的编码后特征 [N, 2*hidden]
        """
        device = x.device
        num_graphs = int(batch_idx.max().item()) + 1
        outs = []
        for i in range(num_graphs):
            mask = (batch_idx == i)
            seq_i = x[mask]              # [Li, F]
            seq_i = seq_i.unsqueeze(0)   # [1, Li, F]
            enc, _ = lstm(seq_i)         # [1, Li, 2H]
            enc = enc.squeeze(0)         # [Li, 2H]

            if attn is not None:
                # residue-level attention，给每个位置一个权重
                score = attn(enc)        # [Li, 1]
                alpha = torch.softmax(score, dim=0)  # [Li,1]
                enc = enc * alpha        # [Li, 2H]  给重要位置放大
            outs.append(enc)
        return torch.cat(outs, dim=0).to(device)

    # ------------------------------------------------------------------
    # 2) 跨模态注意力（Ab↔Ag）
    # ------------------------------------------------------------------
    def _mh_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, heads: int) -> torch.Tensor:
        """
        Q: [Nq, d], K: [Nk, d], V: [Nk, d]
        return: [Nq, d]
        """
        Nq, d = Q.size()
        Nk = K.size(0)
        dh = d // heads

        def split(x):
            return x.view(-1, heads, dh).transpose(0, 1)    # [H, N, dh]

        Qh, Kh, Vh = split(Q), split(K), split(V)
        att = (Qh @ Kh.transpose(-2, -1)) / (dh ** 0.5)     # [H, Nq, Nk]
        att = torch.softmax(att, dim=-1)
        out = att @ Vh                                      # [H, Nq, dh]
        out = out.transpose(0, 1).contiguous().view(Nq, d)  # [Nq, d]
        return out

    def _cross_refine(self, B_z: torch.Tensor, G_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # B ← G
        Qb = self.q_b(B_z); Kg = self.k_g(G_z); Vg = self.v_g(G_z)
        ctx_b = self._mh_attention(Qb, Kg, Vg, self.x_heads)
        B_new = self.o_b(ctx_b)
        B_new = self.x_dp(B_new)
        B_z = self.x_ln_b(B_z + B_new)
        B_z = B_z + self.ffn_b(torch.cat([B_z, ctx_b], dim=-1))

        # G ← B
        Qg = self.q_g(G_z); Kb = self.k_b(B_z); Vb = self.v_b(B_z)
        ctx_g = self._mh_attention(Qg, Kb, Vb, self.x_heads)
        G_new = self.o_g(ctx_g)
        G_new = self.x_dp(G_new)
        G_z = self.x_ln_g(G_z + G_new)
        G_z = G_z + self.ffn_g(torch.cat([G_z, ctx_g], dim=-1))

        return B_z, G_z

     # ------------------------------------------------------------------
    # 2.5 图级 Readout & 对比学习
    # ------------------------------------------------------------------
    def _graph_readout(self, B_z: torch.Tensor, G_z: torch.Tensor,
                       batch: PygBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对每个 Ab/Ag 图做全局 pooling 得到图级表示
        B_z, G_z: [N_nodes, d]
        返回:
          h_b: [num_graphs, d]
          h_g: [num_graphs, d]
        """
        h_b = global_mean_pool(B_z, batch.x_b_batch)  # [num_graphs, d]
        h_g = global_mean_pool(G_z, batch.x_g_batch)  # [num_graphs, d]
        return h_b, h_g

    def graph_cl_loss(self, h_b: torch.Tensor, h_g: torch.Tensor,
                      tau: Optional[float] = None) -> torch.Tensor:
        """
        图级 Ab–Ag 对比学习 InfoNCE / NT-Xent loss
        h_b, h_g: [B, d]（同一行是一个复合物里的 Ab / Ag）
        """
        if tau is None:
            tau = getattr(self, "graph_cl_tau", 0.1)

        # 投影 + L2 normalize
        z_b = self.ab_graph_proj(h_b)   # [B, d_p]
        z_g = self.ag_graph_proj(h_g)   # [B, d_p]

        z_b = F.normalize(z_b, dim=-1)
        z_g = F.normalize(z_g, dim=-1)

        Bsz = z_b.size(0)
        logits = (z_b @ z_g.t()) / tau            # [B, B]
        labels = torch.arange(Bsz, device=logits.device)

        # 双向对比：b->g 和 g->b
        loss_bg = F.cross_entropy(logits, labels)
        loss_gb = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_bg + loss_gb)

    # ------------------------------------------------------------------
    # 3) 解码器
    # ------------------------------------------------------------------
    def _init_decoder(self, decoder: Dict[str, Any]) -> Union[nn.Module, nn.Parameter, None]:
        name = decoder["name"]
        d = self.hparams["dim_list"][-1]
        if name == "bilinear":
            W = nn.Parameter(torch.empty(d, d), requires_grad=True)
            nn.init.kaiming_normal_(W)
            return W
        elif name == "fc":
            hidden = int(decoder.get("hidden", 256))
            dp = float(decoder.get("dropout", 0.1))
            bias = bool(decoder.get("bias", True))
            return nn.Sequential(
                nn.Linear(2*d, hidden, bias=bias),
                nn.GELU(),
                nn.Dropout(dp),
                nn.Linear(hidden, 1, bias=bias),
            )
        elif name == "inner_prod":
            return None
        else:
            raise ValueError(f"Unknown decoder: {name}")

    def _make_decoder_fn(self, decoder: Dict[str, Any]):
        name = decoder["name"]
        if name == "inner_prod":
            return lambda b_z, g_z: b_z @ g_z.t()
        if name == "bilinear":
            return lambda b_z, g_z: b_z @ self.decoder @ g_z.t()
        if name == "fc":
            def runner(b_z: torch.Tensor, g_z: torch.Tensor) -> torch.Tensor:
                Nb, Ng, d = b_z.size(0), g_z.size(0), b_z.size(1)
                chunk = self.decode_chunk_size or (Nb * Ng)
                outs = []
                # 按抗原残基方向切块，避免一次性 Nb×Ng 太大
                per = max(1, int(chunk // max(1, Nb)))
                for s in range(0, Ng, per):
                    g_part = g_z[s:s+per]  # [ng, d]
                    h = torch.cat([
                        b_z.unsqueeze(1).expand(-1, g_part.size(0), -1),
                        g_part.unsqueeze(0).expand(Nb, -1, -1)
                    ], dim=-1)            # [Nb, ng, 2d]
                    out = self.decoder(h).squeeze(-1)  # [Nb, ng]
                    outs.append(out)
                return torch.cat(outs, dim=1)
            return runner
        raise ValueError

    # ------------------------------------------------------------------
    # 4) 生成真值矩阵（和你原版一样的切片方式）
    # ------------------------------------------------------------------
    def _dense_labels(self, batch: PygBatch):
        edge_index_bg_dense = torch.zeros(batch.x_b.shape[0], batch.x_g.shape[0], device=self.device)
        edge_index_bg_dense[batch.edge_index_bg[0], batch.edge_index_bg[1]] = 1
        idx_b = torch.cumsum(torch.cat([torch.zeros(1, device=self.device, dtype=torch.long),
                                        batch.x_b_batch.bincount()]), dim=0)
        idx_g = torch.cumsum(torch.cat([torch.zeros(1, device=self.device, dtype=torch.long),
                                        batch.x_g_batch.bincount()]), dim=0)
        return edge_index_bg_dense, idx_b, idx_g

    # ------------------------------------------------------------------
    # 5) encode / decode / forward
    # ------------------------------------------------------------------
    def encode(self, batch: PygBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = batch.to(self.device)

        # -------- 序列分支：BiLSTM (+ residue-level attention) --------
        seq_b = self._run_seq_encoder(
            batch.x_b, batch.x_b_batch, self.ab_lstm,
            self.ab_seq_attn if self.use_seq_attn else None
        )  # [N_b, 2H]

        seq_g = self._run_seq_encoder(
            batch.x_g, batch.x_g_batch, self.ag_lstm,
            self.ag_seq_attn if self.use_seq_attn else None
        )  # [N_g, 2H]

        # -------- 结构分支：GNN (GCN / GAT / SAGE) --------
        struct_b = self._run_gnn_stack(
            self.B_encoder_block, batch.x_b, batch.edge_index_b
        )  # [N_b, D]
        struct_g = self._run_gnn_stack(
            self.G_encoder_block, batch.x_g, batch.edge_index_g
        )  # [N_g, D]

        # -------- 融合：concat(seq, struct) -> MLP -> D --------
        B_z = self.ab_fuse(torch.cat([seq_b, struct_b], dim=-1))  # [N_b, D]
        G_z = self.ag_fuse(torch.cat([seq_g, struct_g], dim=-1))  # [N_g, D]

        # -------- 跨模态对齐（可选） --------
        if self.use_xattn:
            B_z, G_z = self._cross_refine(B_z, G_z)

        return B_z, G_z

    def decode(self, B_z: torch.Tensor, G_z: torch.Tensor, batch: PygBatch):
        batch = batch.to(self.device)
        pred_list, true_list = [], []

        dense_true, idxb, idxg = self._dense_labels(batch)

        for i in range(batch.num_graphs):
            b_mask = (batch.x_b_batch == i)
            g_mask = (batch.x_g_batch == i)
            b_i = B_z[b_mask]
            g_i = G_z[g_mask]

            scores = self._dc_func(b_i, g_i)     # [Nb_i, Ng_i]
            probs = torch.sigmoid(scores)        # 训练脚本里现在是用的概率版 loss，就保持和你原来一致
            pred_list.append(probs)

            true_i = dense_true[idxb[i]:idxb[i+1], idxg[i]:idxg[i+1]]
            true_list.append(true_i)

        return pred_list, true_list

    def forward(self, batch: PygBatch) -> Dict[str, Union[int, torch.Tensor, List[torch.Tensor]]]:
        batch = batch.to(self.device)
        B_z, G_z = self.encode(batch)
        edge_index_bg_pred, edge_index_bg_true = self.decode(B_z, G_z, batch)

        out = {
            "abdbid": batch.abdbid,
            "edge_index_bg_pred": edge_index_bg_pred,
            "edge_index_bg_true": edge_index_bg_true,
        }

        # >>> 新增：图级表示抛出去，用于 graph-level CL <<<
        if getattr(self, "use_graph_cl", False) and self.use_graph_cl:
            h_b, h_g = self._graph_readout(B_z, G_z, batch)
            out["graph_h_b"] = h_b   # [num_graphs, d]
            out["graph_h_g"] = h_g   # [num_graphs, d]

        return out


# a linear version of the model
class LinearAbAgIntGAE(nn.Module):
    def __init__(
        self,
        input_ab_dim: int,  # input dims
        input_ag_dim: int,  # input dims
        dim_list: List[int],  # dims (length = len(act_list) + 1)
        act_list: List[str],  # acts
        decoder: Optional[Dict] = None,  # layer type
        try_gpu: bool = True,  # use gpu
        input_ab_act: str = "relu",  # input activation
        input_ag_act: str = "relu",  # input activation
    ):
        super().__init__()
        decoder = (
            {
                "name": "inner_prod",
            }
            if decoder is None
            else decoder
        )
        self.device = torch.device(
            "cuda" if try_gpu and torch.cuda.is_available() else "cpu"
        )

        # add to hparams
        self.hparams = {
            "input_ab_dim": input_ab_dim,
            "input_ag_dim": input_ag_dim,
            "dim_list": dim_list,
            "act_list": act_list,
            "decoder": decoder,
        }
        self._args_sanity_check()

        # encoder
        self.B_encoder_block = self._create_a_encoder_block(
            node_feat_name="x_b",
            input_dim=input_ab_dim,
            input_act=input_ab_act,
            dim_list=dim_list,
            act_list=act_list,
        ).to(self.device)
        self.G_encoder_block = self._create_a_encoder_block(
            node_feat_name="x_g",
            input_dim=input_ag_dim,
            input_act=input_ag_act,
            dim_list=dim_list,
            act_list=act_list,
        ).to(self.device)

        # decoder attr placeholder
        self.decoder = self.decoder_factory(self.hparams["decoder"])
        self._dc_func: Callable = self.decoder_func_factory(self.hparams["decoder"])

    def _args_sanity_check(self):
        # 1. if dim_list or act_list is provided, assert dim_list length is equal to act_list length + 1
        if self.hparams["dim_list"] is not None or self.hparams["act_list"] is not None:
            try:
                assert (
                    len(self.hparams["dim_list"]) == len(self.hparams["act_list"]) + 1
                ), (
                    f"dim_list length must be equal to act_list length + 1, "
                    f"got dim_list {self.hparams['dim_list']} and act_list {self.hparams['act_list']}"
                )
            except AssertionError as e:
                raise ValueError(
                    "dim_list length must be equal to act_list length + 1, "
                ) from e
        # 2. if decoder is provided, assert decoder name is in ['inner_prod', 'fc', 'bilinear']
        if self.hparams["decoder"] is not None:
            try:
                assert isinstance(self.hparams["decoder"], Union[dict, DictConfig])
            except AssertionError as e:
                raise TypeError(
                    f"decoder must be a dict, got {self.hparams['decoder']}"
                ) from e
            try:
                assert self.hparams["decoder"]["name"] in (
                    "inner_prod",
                    "fc",
                    "bilinear",
                )
            except AssertionError as e:
                raise ValueError(
                    f"decoder {self.hparams['decoder']['name']} not supported, "
                    "please choose from ['inner_prod', 'fc', 'bilinear']"
                ) from e

    def _create_a_encoder_block(
        self,
        node_feat_name: str,
        input_dim: int,
        input_act: str,
        dim_list: List[int],
        act_list: List[str],
    ):
        def _create_linear_layer(i: int, in_channels: int, out_channels: int) -> tuple:
            if i == 0:
                mapping = f"{node_feat_name} -> {node_feat_name}_{i+1}"
            else:
                mapping = f"{node_feat_name}_{i} -> {node_feat_name}_{i+1}"
            # print(mapping)

            return (
                nn.Linear(in_channels, out_channels),
                mapping,
            )

        def _create_act_layer(act_name: Optional[str]) -> nn.Module:
            # assert act_name is either None or str
            assert act_name is None or isinstance(
                act_name, str
            ), f"act_name must be None or str, got {act_name}"

            if act_name is None:
                return (nn.Identity(),)
            elif act_name.lower() == "relu":
                return (nn.ReLU(inplace=True),)
            elif act_name.lower() == "leakyrelu":
                return (nn.LeakyReLU(inplace=True),)
            else:
                raise ValueError(
                    f"activation {act_name} not supported, please choose from ['relu', 'leakyrelu', None]"
                )

        modules = [
            _create_linear_layer(0, input_dim, dim_list[0]),  # First layer
            _create_act_layer(input_act),
        ]

        for i in range(len(dim_list) - 1):  # Additional layers
            modules.extend(
                [
                    _create_linear_layer(
                        i + 1, dim_list[i], dim_list[i + 1]
                    ),  # i+1 increment due to the input layer
                    _create_act_layer(act_list[i]),
                ]
            )

        return tg.nn.Sequential(input_args=f"{node_feat_name}", modules=modules)

    def _init_fc_decoder(self, decoder) -> nn.Sequential:
        bias: bool = decoder["bias"]
        dp: Optional[float] = decoder["dropout"]

        dc = nn.ModuleList()

        # dropout
        if dp is not None:
            dc.append(nn.Dropout(dp))
        # fc linear
        dc.append(
            nn.Linear(
                in_features=self.hparams["dim_list"][-1] * 2, out_features=1, bias=bias
            )
        )
        # make it a sequential
        dc = nn.Sequential(*dc)

        return dc

    def encode(self, batch: PygBatch) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch: (PygBatch) batched data returned by PyG DataLoader
        Returns:
            B_z: (Tensor) shape (Nb, C)
            G_z: (Tensor) shape (Ng, C)
        """
        batch = batch.to(self.device)
        B_z = self.B_encoder_block(batch.x_b)  # , batch.edge_index_b)  # (Nb, C)
        G_z = self.G_encoder_block(batch.x_g)  # , batch.edge_index_g)  # (Ng, C)

        return B_z, G_z

    def decoder_factory(
        self, decoder_dict: Dict[str, str]
    ) -> Union[nn.Module, nn.Parameter, None]:
        name = decoder_dict["name"]

        if name == "bilinear":
            init_method = decoder_dict.get("init_method", "kaiming_normal_")
            decoder = nn.Parameter(
                data=torch.empty(
                    self.hparams["dim_list"][-1], self.hparams["dim_list"][-1]
                ),
                requires_grad=True,
            )
            torch.nn.init.__dict__[init_method](decoder)
            return decoder

        elif name == "fc":
            return self._init_fc_decoder(decoder_dict)

        elif name == "inner_prod":
            return

    def decoder_func_factory(self, decoder_dict: Dict[str, str]) -> Callable:
        name = decoder_dict["name"]

        if name == "bilinear":
            return lambda b_z, g_z: b_z @ self.decoder @ g_z.t()

        elif name == "fc":

            def _fc_runner(b_z: Tensor, g_z: Tensor) -> Tensor:
                """
                # (Nb, Ng, C*2)) -> (Nb, Ng, 1)
                # h = torch.cat([
                #         z_ab.unsqueeze(1).tile((1, z_ag.size(0), 1)),  # (Nb, 1, C) -> (Nb, Ng, C)
                #         z_ag.unsqueeze(0).tile((z_ab.size(0), 1, 1)),  # (1, Ng, C) -> (Nb, Ng, C)
                #     ], dim=-1)
                """
                h = torch.cat(
                    [
                        b_z.unsqueeze(1).expand(
                            -1, g_z.size(0), -1
                        ),  # (Nb, 1, C) -> (Nb, Ng, C)
                        g_z.unsqueeze(0).expand(
                            b_z.size(0), -1, -1
                        ),  # (1, Ng, C) -> (Nb, Ng, C)
                    ],
                    dim=-1,
                )
                # (Nb, Ng, C*2) -> (Nb, Ng, 1)
                h = self.decoder(h)
                return h.squeeze(-1)  # (Nb, Ng, 1) -> (Nb, Ng)

            return _fc_runner

        elif name == "inner_prod":
            return lambda b_z, g_z: b_z @ g_z.t()

    def decode(
        self, B_z: Tensor, G_z: Tensor, batch: PygBatch
    ) -> Tuple[Tensor, Tensor]:
        """
        Inner Product Decoder

        Args:
            z_ab: (Tensor)  shape (Nb, dim_latent)
            z_ag: (Tensor)  shape (Ng, dim_latent)

        Returns:
            A_reconstruct: (Tensor) shape (B, G)
                reconstructed bipartite adjacency matrix
        """
        # move batch to device
        batch = batch.to(self.device)

        edge_index_bg_pred = []
        edge_index_bg_true = []

        # dense bipartite edge index
        edge_index_bg_dense = torch.zeros(batch.x_b.shape[0], batch.x_g.shape[0]).to(
            self.device
        )
        edge_index_bg_dense[batch.edge_index_bg[0], batch.edge_index_bg[1]] = 1

        # get graph sizes (number of nodes) in the batch, used to slice the dense bipartite edge index
        node2graph_idx = torch.stack(
            [
                torch.cumsum(
                    torch.cat(
                        [
                            torch.zeros(1).long().to(self.device),
                            batch.x_b_batch.bincount(),
                        ]
                    ),
                    dim=0,
                ),  # (Nb+1, ) CDR     nodes
                torch.cumsum(
                    torch.cat(
                        [
                            torch.zeros(1).long().to(self.device),
                            batch.x_g_batch.bincount(),
                        ]
                    ),
                    dim=0,
                ),  # (Ng+1, ) antigen nodes
            ],
            dim=0,
        )

        for i in range(batch.num_graphs):
            edge_index_bg_pred.append(
                F.sigmoid(
                    self._dc_func(
                        b_z=B_z[batch.x_b_batch == i], g_z=G_z[batch.x_g_batch == i]
                    )
                )
            )  # Tensor (Nb, Ng)
            edge_index_bg_true.append(
                edge_index_bg_dense[
                    node2graph_idx[0, i] : node2graph_idx[0, i + 1],
                    node2graph_idx[1, i] : node2graph_idx[1, i + 1],
                ]
            )  # Tensor (Nb, Ng)

        return edge_index_bg_pred, edge_index_bg_true

    def forward(self, batch: PygBatch) -> Dict[str, Union[int, Tensor]]:
        # device
        batch = batch.to(self.device)
        # encode
        z_ab, z_ag = self.encode(batch)  # (Nb, C), (Ng, C)
        # decode
        edge_index_bg_pred, edge_index_bg_true = self.decode(z_ab, z_ag, batch)

        return {
            "abdbid": batch.abdbid,  # List[str]
            "edge_index_bg_pred": edge_index_bg_pred,  # List[Tensor (Nb, Ng)]
            "edge_index_bg_true": edge_index_bg_true,  # List[Tensor (Nb, Ng)]
        }
