from typing import Dict

import torch
from torch import Tensor
from torcheval.metrics import BinaryAUPRC, BinaryConfusionMatrix, BinaryAUROC
from torcheval.metrics.functional import binary_auprc
from torchmetrics.functional import matthews_corrcoef
from typing import Optional


def cal_edge_index_bg_auprc(
    edge_index_bg_pred: Tensor,
    edge_index_bg_true: Tensor,
    edge_cutoff: float = 0.5,
) -> Tensor:
    with torch.no_grad():
        t = edge_index_bg_true.reshape(-1).long().cpu()
        p = (edge_index_bg_pred > edge_cutoff).reshape(-1).long().cpu()

        return binary_auprc(p, t)


def cal_epitope_node_auprc(
    edge_index_bg_pred: Tensor,
    edge_index_bg_true: Tensor,
    num_edge_cutoff: int,  # used to determine epitope residue from edges,
) -> Tensor:
    with torch.no_grad():
        # get epitope idx
        t = (edge_index_bg_true.sum(dim=0) > 0).reshape(-1).long()
        p = (edge_index_bg_pred.sum(dim=0) > num_edge_cutoff).reshape(-1).long()

        return binary_auprc(p, t)

def _zero_metrics():
    zero = torch.tensor(0.0)
    return {
        "tn": zero,
        "fp": zero,
        "fn": zero,
        "tp": zero,
        "auprc": zero,
        "mcc": zero,
        "best_thr": torch.tensor(0.5),
        "precision": zero,
        "recall": zero,
        "f1": zero,
        "n_samples": zero,
        "pred": torch.empty(0),
        "true": torch.empty(0, dtype=torch.long),
    }


def cal_epitope_node_metrics(
    edge_index_bg_pred: Tensor,
    edge_index_bg_true: Tensor,
    fixed_threshold: Optional[float] = None,
):
    """
    Epitope 残基级别指标（支持单图 [Nb, Ng] 和多图 [B, Nb, Ng]）

    - 输入：
        edge_index_bg_pred: 预测边概率矩阵，形状 [Nb, Ng] 或 [B, Nb, Ng]
        edge_index_bg_true: 真实边 0/1 矩阵，形状同上
    - 逻辑：
        对每一张图：
          t_b: 某个抗原残基是否至少有一条真实接触边
          s_b: 该残基所有预测边概率之和（归一化到 [0,1]）
        把所有图的 (t_b, s_b) 串起来，统一算 AUPRC / MCC（扫阈值）
    """
    with torch.no_grad():
        if edge_index_bg_true.numel() == 0:
            return _zero_metrics()

        # --------- 1. 统一成 batch 维度 ---------
        if edge_index_bg_true.dim() == 2:
            # 单图: [Nb, Ng] -> [1, Nb, Ng]
            edge_index_bg_true_b = edge_index_bg_true.unsqueeze(0)
            edge_index_bg_pred_b = edge_index_bg_pred.unsqueeze(0)
        elif edge_index_bg_true.dim() == 3:
            # 多图: [B, Nb, Ng]
            edge_index_bg_true_b = edge_index_bg_true
            edge_index_bg_pred_b = edge_index_bg_pred
        else:
            raise ValueError(
                f"edge_index_bg_true 期望是 2D 或 3D 张量，得到 shape = {edge_index_bg_true.shape}"
            )

        B = edge_index_bg_true_b.shape[0]

        all_t = []
        all_s = []

        # --------- 2. 对每张图计算节点分数 ---------
        for b in range(B):
            true_mat = edge_index_bg_true_b[b]  # [Nb, Ng]
            pred_mat = edge_index_bg_pred_b[b]  # [Nb, Ng]

            # 每个 Ag 残基是否为 epitope: 该列是否有真实边
            t_b = (true_mat.sum(dim=0) > 0).reshape(-1).long().cpu()  # [Ng]

            if t_b.numel() == 0:
                continue  # 这张图没有 Ag 节点，跳过

            # 残基连续得分：该列所有预测边概率之和
            node_score_b = pred_mat.sum(dim=0).reshape(-1)  # [Ng]
            max_val = node_score_b.max()

            if max_val <= 0:
                s_b = torch.zeros_like(node_score_b)
            else:
                s_b = node_score_b / (max_val + 1e-7)

            all_t.append(t_b)
            all_s.append(s_b.cpu())

        # 所有图都没有效节点的情况
        if len(all_t) == 0:
            return _zero_metrics()

        # 拼接所有图的残基
        t = torch.cat(all_t, dim=0)         # [N_all]
        s_cpu = torch.cat(all_s, dim=0)     # [N_all]

        # --------- 3. AUPRC（用连续得分） ---------
        auprc = BinaryAUPRC().update(input=s_cpu, target=t).compute()
        auroc = BinaryAUROC().update(input=s_cpu, target=t).compute()

        # --------- 4. 扫阈值，选 MCC 最大 ---------
        thresholds = torch.arange(0.01, 1.0, 0.01)
        best_mcc = torch.tensor(-1.0)
        best_thr = torch.tensor(0.5)
        best_cm = None

        if fixed_threshold is None:
            for thr in thresholds:
                p_bin = (s_cpu >= thr).long()
                tn, fp, fn, tp = (
                    BinaryConfusionMatrix()
                    .update(input=p_bin, target=t)
                    .compute()
                    .reshape(-1)
                )

                denom = torch.sqrt(
                    (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-7
                )
                mcc = (tp * tn - fp * fn) / denom

                if torch.isfinite(mcc) and mcc > best_mcc:
                    best_mcc = mcc
                    best_thr = thr
                    best_cm = (tn, fp, fn, tp)

            if best_cm is None:
                return _zero_metrics()
        else:
            best_thr = fixed_threshold
            p_bin = (s_cpu >= best_thr).long()
            tn, fp, fn, tp = (
                BinaryConfusionMatrix()
                .update(input=p_bin, target=t)
                .compute()
                .reshape(-1)
            )
            denom = torch.sqrt(
                (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-7
            )
            mcc = (tp * tn - fp * fn) / denom

            best_mcc = mcc
            best_cm = (tn, fp, fn, tp)

        tn, fp, fn, tp = best_cm

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        n_samples = (tn + fp + fn + tp).float()

        return {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "auprc": auprc,
            "auroc": auroc,
            "mcc": best_mcc,
            "best_thr": best_thr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_samples": n_samples,
            "pred": s_cpu,   # 所有图拼在一起后的残基分数
            "true": t,       # 所有图拼在一起后的 0/1 标签
        }

def cal_edge_index_bg_metrics(
    edge_index_bg_pred: torch.Tensor,
    edge_index_bg_true: torch.Tensor,
    edge_cutoff: float = 0.5,
) -> dict:
    """
    边级指标（edge-level），支持：

      - 单图: [Nb, Ng]
      - 多图: [B, Nb, Ng]

    做法：把所有图的边 flatten 到一起，统一算 AUPRC 和扫阈值的 MCC。
    返回的 n_samples = 所有图的“边数总和”（= tn+fp+fn+tp）。
    """
    with torch.no_grad():
        # 如果完全没有数据，直接返回 0
        if edge_index_bg_true.numel() == 0:
            return _zero_edge_metrics(edge_cutoff)

        # -------- 1. 统一成带 batch 的形状 --------
        if edge_index_bg_true.dim() == 2:
            # 单图: [Nb, Ng] -> [1, Nb, Ng]
            true_b = edge_index_bg_true.unsqueeze(0)
            pred_b = edge_index_bg_pred.unsqueeze(0)
        elif edge_index_bg_true.dim() == 3:
            # 多图: [B, Nb, Ng]
            true_b = edge_index_bg_true
            pred_b = edge_index_bg_pred
        else:
            raise ValueError(
                f"edge_index_bg_true 期望是 2D 或 3D 张量，当前 shape = {edge_index_bg_true.shape}"
            )

        # -------- 2. flatten 所有图的边 --------
        t = true_b.reshape(-1).long().cpu()  # [N_edges_all]
        p = pred_b.reshape(-1).cpu()         # [N_edges_all]

        # 如果 flatten 后还是没有样本（极端情况），返回 0
        if t.numel() == 0:
            return _zero_edge_metrics(edge_cutoff)

        # === 3. AUPRC（连续分数） ===
        auprc = BinaryAUPRC().update(input=p, target=t).compute()

        # === 4. 扫描不同阈值，找 MCC 最大点 ===
        # 保持你原来的 0.01 ~ 0.7，如果想和 node-level 对齐可以改成 torch.arange(0.01, 1.0, 0.01)
        thresholds = torch.arange(0.01, 0.7, 0.01)
        best_mcc = torch.tensor(-1.0)
        best_thr = torch.tensor(edge_cutoff)

        for thr in thresholds:
            tn, fp, fn, tp = (
                BinaryConfusionMatrix(threshold=thr.item())
                .update(input=p, target=t)
                .compute()
                .reshape(-1)
            )

            denom = torch.sqrt(
                (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-7
            )
            mcc = (tp * tn - fp * fn) / denom

            if torch.isfinite(mcc) and mcc > best_mcc:
                best_mcc = mcc
                best_thr = thr

        # === 5. 在最佳阈值处再计算一次混淆矩阵 ===
        tn, fp, fn, tp = (
            BinaryConfusionMatrix(threshold=best_thr.item())
            .update(input=p, target=t)
            .compute()
            .reshape(-1)
        )

        # === 6. precision / recall / F1 ===
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        n_samples = (tn + fp + fn + tp).float()

        return {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "auprc": auprc,
            "mcc": best_mcc,
            "best_thr": best_thr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_samples": n_samples,
        }