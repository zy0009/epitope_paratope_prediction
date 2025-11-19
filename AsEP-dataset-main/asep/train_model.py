import os
import os.path as osp
from pathlib import Path
from pprint import pformat, pprint
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch import Tensor
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch as PygBatch
from torch_geometric.loader import DataLoader as PygDataLoader
from tqdm import tqdm

# custom
from asep.data.asepv1_dataset import AsEPv1Dataset
from asep.data.embedding.handle import EmbeddingHandler
from asep.data.embedding_config import EmbeddingConfig
from asep.model import loss as loss_module
from asep.model.asepv1_model import LinearAbAgIntGAE, PyGAbAgIntGAE
from asep.model.callbacks import EarlyStopper, ModelCheckpoint
from asep.model.metric import (cal_edge_index_bg_metrics,
                               cal_epitope_node_metrics)
from asep.model.utils import generate_random_seed, seed_everything
from asep.utils import time_stamp

# ==================== Configuration ====================
# set precision
torch.set_float32_matmul_precision("high")

ESM2DIM = {
    "esm2_t6_8M_UR50D": 320,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t33_650M_UR50D": 1280,
}

DataRoot = Path.cwd().joinpath("data")


# ==================== Function ====================
# PREPARE: EmbeddingConfig
def create_embedding_config(dataset_config: Dict[str, Any]) -> EmbeddingConfig:
    """
    Create embedding config from config dict

    Args:
        dataset_config (Dict[str, Any]): dataset config

    Returns:
        EmbeddingConfig: embedding config
    """
    # assert dataset_config is a primitive dict
    try:
        assert isinstance(dataset_config, dict)
    except AssertionError as e:
        raise TypeError(f"dataset_config must be a dict, instead got {type(dataset_config)}") from e

    if dataset_config["node_feat_type"] in ("pre_cal", "one_hot"):
        # parse the embedding model for ab and ag
        d = dict(
            node_feat_type=dataset_config["node_feat_type"],
            ab=dataset_config["ab"].copy(),
            ag=dataset_config["ag"].copy(),
        )
        return EmbeddingConfig(**d)

    # otherwise, node_feat_type is custom, need to load function from user specified script
    try:
        d = dataset_config["ab"]["custom_embedding_method_src"]
        ab_func = EmbeddingHandler(
            script_path=d["script_path"], function_name=d["method_name"]
        ).embed
    except Exception as e:
        raise RuntimeError(
            "Error loading custom embedding method for Ab. Please check the script."
        ) from e
    try:
        d = dataset_config["ag"]["custom_embedding_method_src"]
        ag_func = EmbeddingHandler(
            script_path=d["script_path"], function_name=d["method_name"]
        ).embed
    except Exception as e:
        raise RuntimeError(
            "Error loading custom embedding method for Ag. Please check the script."
        ) from e
    updated_dataset_config = dataset_config.copy()
    updated_dataset_config["ab"]["custom_embedding_method"] = ab_func
    updated_dataset_config["ag"]["custom_embedding_method"] = ag_func
    return EmbeddingConfig(** updated_dataset_config)


# PREPARE: dataset
def create_asepv1_dataset(
    root: str = None,
    name: str = None,
    embedding_config: EmbeddingConfig = None,
):
    """
    Create AsEPv1 dataset

    Args:
        root (str, optional): root directory for dataset. Defaults to None.
            if None, set to './data'
        name (str, optional): dataset name. Defaults to None.
            if None, set to 'asep'
        embedding_config (EmbeddingConfig, optional): embedding config. Defaults to None.
            if None, use default embedding config
            {
                'node_feat_type': 'pre_cal',
                'ab': {'embedding_model': 'igfold'},
                'ag': {'embedding_model': 'esm2'},
            }

    Returns:
        AsEPv1Dataset: AsEPv1 dataset
    """
    root = root if root is not None else "./data"
    embedding_config = embedding_config or EmbeddingConfig()
    asepv1_dataset = AsEPv1Dataset(
        root=root, name=name, embedding_config=embedding_config
    )
    return asepv1_dataset


# PREPARE: dataloaders
def create_asepv1_dataloaders(
    asepv1_dataset: AsEPv1Dataset,
    config: Dict[str, Any] = None,
    split_method: str = None,
    split_idx: Dict[str, Tensor] = None,
    return_dataset: bool = False,
    dev: bool = False,
) -> Tuple[PygDataLoader, PygDataLoader, PygDataLoader]:
    """
    Create dataloaders for AsEPv1 dataset

    Args:
        config (Dict[str, Any], optional): config dict. Defaults to None.
        return_dataset (bool, optional): return dataset instead of dataloaders. Defaults to False.
        dev (bool, optional): use dev mode. Defaults to False.
        split_idx (Dict[str, Tensor], optional): split index. Defaults to None.
    AsEPv1Dataset kwargs:
        embedding_config (EmbeddingConfig, optional): embedding config. Defaults to None.
            If None, use default EmbeddingConfig, for details, see asep.data.embedding_config.EmbeddingConfig.
        split_method (str, optional): split method. Defaults to None. Either 'epitope_ratio' or 'epitope_group'

    Returns:
        Tuple[PygDataLoader, PygDataLoader, PygDataLoader]: train/val/test dataloaders
    """
    # split dataset
    split_idx = split_idx or asepv1_dataset.get_idx_split(split_method=split_method)
    train_set = asepv1_dataset[split_idx["train"]]
    val_set = asepv1_dataset[split_idx["val"]]
    test_set = asepv1_dataset[split_idx["test"]]

    # if dev, only use 170 train samples (keep val/test as is for quick validation)
    if dev:
        train_set = train_set[:170]
        logger.info(f"Dev mode enabled: train set truncated to {len(train_set)} samples")

    # patch: if test_batch_size is not specified, use val_batch_size
    if ("test_batch_size" not in config["hparams"].keys()) or (
        config["hparams"]["test_batch_size"] is None
    ):
        config["hparams"]["test_batch_size"] = config["hparams"]["val_batch_size"]
        logger.warning(
            f"test_batch_size not specified, using val_batch_size: {config['hparams']['test_batch_size']}"
        )

    # Dataloader common settings (follow PyG batch conventions for graph data)
    _default_kwargs = dict(follow_batch=["x_b", "x_g"])
    _default_kwargs_train = dict(
        batch_size=config["hparams"]["train_batch_size"], 
        shuffle=True,  # Shuffle train set for better training
        **_default_kwargs
    )
    _default_kwargs_val = dict(
        batch_size=config["hparams"]["val_batch_size"],** _default_kwargs
    )
    _default_kwargs_test = dict(
        batch_size=config["hparams"]["test_batch_size"], **_default_kwargs
    )

    train_loader = PygDataLoader(train_set,** _default_kwargs_train)
    val_loader = PygDataLoader(val_set, **_default_kwargs_val)
    test_loader = PygDataLoader(test_set,** _default_kwargs_test)

    # Log dataset sizes (no wandb, use logger instead)
    logger.info(f"Dataset split completed:")
    logger.info(f"Train samples: {len(train_set)} (batches: {len(train_loader)})")
    logger.info(f"Val samples: {len(val_set)} (batches: {len(val_loader)})")
    logger.info(f"Test samples: {len(test_set)} (batches: {len(test_loader)})")

    if not return_dataset:
        return train_loader, val_loader, test_loader
    return train_set, val_set, test_set, train_loader, val_loader, test_loader


# PREPARE: model
def create_model(
    config: Dict[str, Any]
) -> nn.Module:
    """
    Create model based on config (removed wandb watch)

    Args:
        config (Dict[str, Any]): config dict with model settings

    Returns:
        nn.Module: initialized model
    """
    # Select model architecture
    if config["hparams"]["model_type"] == "linear":
        model_architecture = LinearAbAgIntGAE
    elif config["hparams"]["model_type"] == "graph":
        model_architecture = PyGAbAgIntGAE
    else:
        raise ValueError(f"model_type must be 'linear' or 'graph', got {config['hparams']['model_type']}")

    # Initialize model with config parameters
    model = model_architecture(
        input_ab_dim=config["hparams"]["input_ab_dim"],
        input_ag_dim=config["hparams"]["input_ag_dim"],
        input_ab_act=config["hparams"]["input_ab_act"],
        input_ag_act=config["hparams"]["input_ag_act"],
        dim_list=config["hparams"]["dim_list"],
        act_list=config["hparams"]["act_list"],
        decoder=config["hparams"]["decoder"],
        try_gpu=config["try_gpu"],
        seq_enc=config["hparams"]["seq_enc"],
        encoder=config["hparams"]["encoder"],
        cross_attention=config["hparams"]["cross_attention"],
    )

    # Log model architecture
    logger.info(f"Model initialized (type: {config['hparams']['model_type']}):")
    logger.info(model)
    return model


# PREPARE: loss callables
def generate_loss_callables_from_config(
    loss_config: Dict[str, Any],
) -> List[Tuple[str, Callable, Tensor, Dict[str, Any]]]:
    """
    Generate loss functions with weights from config

    Args:
        loss_config (Dict[str, Any]): config for loss terms (name, weight, kwargs)

    Returns:
        List[Tuple]: list of (loss_name, loss_fn, loss_weight, loss_kwargs)
    """
    # Validate each loss term has required keys
    for loss_name, kwargs in loss_config.items():
        try:
            assert "name" in kwargs.keys() and "w" in kwargs.keys()
        except AssertionError as e:
            raise KeyError(f"Loss term '{loss_name}' must contain keys 'name' (loss function name) and 'w' (weight)") from e

    # Create loss callables list
    loss_callables: List[Tuple[str, Callable, Tensor, Dict[str, Any]]] = [
        (
            loss_name,  # Unique name for the loss term
            getattr(loss_module, kwargs["name"]),  # Get loss function from loss module
            torch.tensor(kwargs["w"], dtype=torch.float32),  # Loss weight (cast to float32)
            kwargs.get("kwargs", {}),  # Additional kwargs for loss function
        )
        for loss_name, kwargs in loss_config.items()
    ]

    # Log loss configuration
    logger.info("Loss functions initialized:")
    for name, fn, w, kwargs in loss_callables:
        logger.info(f"- {name}: function={fn.__name__}, weight={w.item()}, kwargs={kwargs}")
    return loss_callables


def feed_forward_step(
    model: nn.Module,
    batch: PygBatch,
    loss_callables: List[Tuple[str, Callable, Tensor, Dict]],
    is_train: bool,
    edge_cutoff: Optional[int] = None,
    num_edge_cutoff: Optional[int] = None,
) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
    """
    Feed forward and calculate loss & metrics for a batch of AbAg graph pairs

    Returns:
        Tuple[Tensor, Dict, Dict]: average batch loss, edge metrics, epitope node metrics
        （这里后两个现在你还没用，可以后面再补 metrics）
    """
    # Set model mode (train/eval)
    model.train() if is_train else model.eval()

    # Feed forward (no gradient computation for eval)
    with torch.set_grad_enabled(is_train):
        batch_result = model(batch)
        edge_index_bg_pred = batch_result["edge_index_bg_pred"]
        edge_index_bg_true = batch_result["edge_index_bg_true"]

        # ----------------------
        # 1. 边级/对数似然相关 loss
        # ----------------------
        batch_loss = None
        for loss_name, loss_fn, loss_w, loss_kwargs in loss_callables:
            if loss_name == "edge_index_bg_rec_loss":
                # 针对每个 graph-pair 分别算重建损失，再 stack
                loss_v = torch.stack(
                    [loss_fn(x, y, **loss_kwargs) for x, y in zip(edge_index_bg_pred, edge_index_bg_true)]
                )  # [B]
            elif loss_name == "edge_index_bg_sum_loss":
                # 只依赖预测矩阵本身的 loss（比如 L1、L2 正则之类）
                loss_v = torch.stack([loss_fn(x, **loss_kwargs) for x in edge_index_bg_pred])  # [B]
            else:
                raise ValueError(f"Unsupported loss name: {loss_name}")

            weighted_loss = loss_v * loss_w  # [B]
            batch_loss = weighted_loss if batch_loss is None else batch_loss + weighted_loss  # [B]

        # ----------------------
        # 2. 原来就有的 CL loss（local-global / node-level 等）
        #    这里保持你的写法：batch_result["cl_loss"] 通常是一个标量
        # ----------------------
        if "cl_loss" in batch_result and batch_result["cl_loss"] is not None:
            cl_loss = batch_result["cl_loss"]  # 标量
            if batch_loss is None:
                # 理论上不会走到这里，只是防御
                batch_loss = cl_loss
            else:
                batch_loss = batch_loss + cl_loss  # 标量会 broadcast 到每个 graph

        # ----------------------
        # 3. 新增：图级对比学习 graph-level CL loss
        # ----------------------
        graph_cl_loss = None
        if getattr(model, "use_graph_cl", False) and model.use_graph_cl \
           and "graph_h_b" in batch_result and "graph_h_g" in batch_result:

            # print("112 113")
            h_b = batch_result["graph_h_b"]  # [B, d]
            h_g = batch_result["graph_h_g"]  # [B, d]

            # 模型内部实现的 InfoNCE / NT-Xent
            graph_cl_loss = model.graph_cl_loss(h_b, h_g)  # 标量

            # 从 hparams 里读 weight，没有就默认 0.1
            graph_cfg = getattr(model, "hparams", {}).get("graph_cl", {})
            lambda_graph_cl = float(graph_cfg.get("weight", 0.1))
            # print(f"Batch_Loss:{batch_loss.mean()}  CL_Loss: {graph_cl_loss}")

            if batch_loss is None:
                batch_loss = lambda_graph_cl * graph_cl_loss
            else:
                batch_loss = batch_loss + lambda_graph_cl * graph_cl_loss  # 标量同样会 broadcast

        # ----------------------
        # 4. batch 内平均
        # ----------------------
        # 此时 batch_loss 形状：
        #   - 如果只有边级 loss： [B]
        #   - 加了若干标量 CL loss：仍然是 [B] + 标量广播
        avg_loss = batch_loss.mean()

        # 目前你还没在这里统计 metrics，就先返回 edge_index_* 让外面去算
        return avg_loss, edge_index_bg_pred, edge_index_bg_true


def epoch_end(
    preds: Any,
    trues: Any,
    fixed_threshold: Optional[float] = None
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """
    计算一个 epoch 内的全局 edge / epitope 指标。

    参数可以是任意嵌套结构，但最底层必须是 (Tensor, Tensor) 对：
        - Tensor
        - List[Tensor]
        - List[List[Tensor]]
        - 甚至更深层的 List[...] / Tuple[...] 组合

    最底层 Tensor 的形状支持：
        - [Nb, Ng]       单图 / 单 batch
        - [B_b, Nb, Ng]  一个 batch 里多图

    做法：
        1. 递归遍历 (preds, trues)，收集所有 (p, t) Tensor 对。
        2. 对每对 (p, t) 调用 cal_edge_index_bg_metrics / cal_epitope_node_metrics。
        3. 按各自的 n_samples 做加权平均（只对标量指标），
           epi 的 pred/true 则直接 cat，得到全 epoch 的向量。
    """

    with torch.no_grad():

        # -------- 1. 递归收集所有 (pred, true) Tensor 对 --------
        pairs: List[Tuple[Tensor, Tensor]] = []

        def collect_pairs(p_obj: Any, t_obj: Any):
            # 两边都是 Tensor -> 收集
            if isinstance(p_obj, Tensor) and isinstance(t_obj, Tensor):
                pairs.append((p_obj, t_obj))
                return

            # 两边都是 list/tuple -> 同结构递归
            if isinstance(p_obj, (list, tuple)) and isinstance(t_obj, (list, tuple)):
                if len(p_obj) != len(t_obj):
                    raise ValueError("preds 和 trues 的嵌套结构长度不一致")
                for sub_p, sub_t in zip(p_obj, t_obj):
                    collect_pairs(sub_p, sub_t)
                return

            # 其他情况说明结构不匹配
            raise TypeError(
                f"pred 和 true 的结构不匹配或含有非 Tensor/list/tuple 类型: "
                f"type(pred)={type(p_obj)}, type(true)={type(t_obj)}"
            )

        collect_pairs(preds, trues)

        if len(pairs) == 0:
            raise ValueError("在 preds / trues 中没有找到任何 (Tensor, Tensor) 对")


        # -------- 2. 对每一对 (p, t) 计算 batch 级 metrics --------
        all_edge_metrics: List[Dict[str, Tensor]] = []
        all_epi_metrics: List[Dict[str, Tensor]]  = []

        for p, t in pairs:
            edge_m = cal_edge_index_bg_metrics(
                edge_index_bg_pred=p,
                edge_index_bg_true=t,
                edge_cutoff=0.5,
            )
            epi_m = cal_epitope_node_metrics(
                edge_index_bg_pred=p,
                edge_index_bg_true=t,
                fixed_threshold=fixed_threshold
            )
            all_edge_metrics.append(edge_m)
            all_epi_metrics.append(epi_m)

        # -------- 一个小工具：对 metric list 做加权平均（只处理标量） --------
        def weighted_avg_metrics(
            metrics_list: List[Dict[str, Tensor]],
            skip_keys: List[str],
        ) -> Dict[str, Tensor]:
            weights = torch.stack([m["n_samples"] for m in metrics_list]).float()
            weight_sum = weights.sum().clamp_min(1.0)

            out: Dict[str, Tensor] = {}
            for k in metrics_list[0].keys():
                if k in skip_keys:
                    continue
                v0 = metrics_list[0][k]
                # 只对「标量」做加权平均：numel == 1
                if v0.numel() != 1:
                    continue
                vals = torch.stack([
                    m[k].float() * w for m, w in zip(metrics_list, weights)
                ])
                out[k] = vals.sum() / weight_sum

            out["n_samples"] = weight_sum
            return out

        # -------- 3. edge-level：按 n_samples 加权平均（全是标量，很干净） --------
        avg_edge_metrics = weighted_avg_metrics(
            all_edge_metrics,
            skip_keys=["n_samples"],   # edge 里本来就没 pred/true
        )

        # -------- 4. epi-level：加权平均 + 拼接 pred/true --------
        # 4.1 先对标量做加权平均
        avg_epi_metrics = weighted_avg_metrics(
            all_epi_metrics,
            skip_keys=["n_samples", "pred", "true", "best_thr"],  # pred/true 不参与加权
        )

        # 4.2 再把每个 batch 的 pred/true cat 起来，得到全 epoch 的向量
        if "pred" in all_epi_metrics[0] and "true" in all_epi_metrics[0]:
            all_pred_vec = torch.cat([m["pred"] for m in all_epi_metrics], dim=0)
            all_true_vec = torch.cat([m["true"] for m in all_epi_metrics], dim=0)
            avg_epi_metrics["pred"] = all_pred_vec
            avg_epi_metrics["true"] = all_true_vec

        avg_epi_metrics["best_thr"] = epi_m["best_thr"]
        return avg_edge_metrics, avg_epi_metrics

# TRAIN helper - learning rate scheduler
def exec_lr_scheduler(
    ck_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    config: Dict[str, Any],
    val_epoch_metrics: Dict[str, Tensor],
) -> None:
    """
    Execute learning rate scheduler step (supports ReduceLROnPlateau and standard schedulers)

    Args:
        ck_lr_scheduler (Optional[_LRScheduler]): initialized LR scheduler
        config (Dict[str, Any]): config dict with scheduler settings
        val_epoch_metrics (Dict[str, Tensor]): validation metrics for plateau scheduler
    """
    if ck_lr_scheduler is not None:
        # Handle ReduceLROnPlateau which requires metrics
        if config["callbacks"]["lr_scheduler"]["name"] == "ReduceLROnPlateau" and \
           config["callbacks"]["lr_scheduler"]["step"] is not None:
            metric_name = config["callbacks"]["lr_scheduler"]["step"]["metrics"]
            ck_lr_scheduler.step(metrics=val_epoch_metrics[metric_name])
        else:
            # Standard scheduler (step without metrics)
            ck_lr_scheduler.step()


# MAIN function
def train_model(
    config: Dict,
    tb_writer: Optional[SummaryWriter] = None,
):
    """
    Main training function (removed all wandb references)

    Args:
        config: (Dict) config dict, contains all hyperparameters
        tb_writer: (SummaryWriter, optional) TensorBoard writer for logging
    """
    # Set number of threads for PyTorch
    torch.set_num_threads(config["num_threads"])
    logger.info(f"Using {config['num_threads']} threads for PyTorch")

    # --------------------
    # Datasets & Dataloaders
    # --------------------
    dev = config.get("mode") == "dev"
    embedding_config = create_embedding_config(dataset_config=config["dataset"])
    asepv1_dataset = create_asepv1_dataset(
        root=config["dataset"]["root"],
        name=config["dataset"]["name"],
        embedding_config=embedding_config,
    )

    train_loader, val_loader, test_loader = create_asepv1_dataloaders(
        asepv1_dataset=asepv1_dataset,
        config=config,
        split_idx=config["dataset"]["split_idx"],
        split_method=config["dataset"]["split_method"],
        dev=dev,
    )

    # --------------------
    # Model, Loss, Optimizer & Callbacks
    # --------------------
    model = create_model(config=config)

    # Loss functions
    loss_callables = generate_loss_callables_from_config(config["loss"])

    # Optimizer
    optimizer = getattr(torch.optim, config["optimizer"]["name"])(
        params=model.parameters(),
        **config["optimizer"]["params"],
    )
    logger.info(f"Optimizer initialized: {config['optimizer']['name']} with params {config['optimizer']['params']}")

    # Callbacks
    ck_early_stop = (
        EarlyStopper(** config["callbacks"]["early_stopping"])
        if config["callbacks"]["early_stopping"] is not None
        else None
    )
    ck_model_ckpt = (
        ModelCheckpoint(**config["callbacks"]["model_checkpoint"])
        if config["callbacks"]["model_checkpoint"] is not None
        else None
    )
    ck_model_ckpt_edge = (
        ModelCheckpoint(** config["callbacks"]["model_checkpoint_edge"])
        if config["callbacks"]["model_checkpoint_edge"] is not None
        else None
    )
    ck_lr_scheduler = (
        getattr(lr_scheduler, config["callbacks"]["lr_scheduler"]["name"])(
            optimizer=optimizer, **config["callbacks"]["lr_scheduler"]["kwargs"]
        )
        if config["callbacks"]["lr_scheduler"] is not None
        else None
    )

    # --------------------
    # Train Val Test Loop
    # --------------------
    current_epoch_idx, current_val_metric = None, None
    
    train_pred,train_true =  [],[]
    for epoch_idx in range(config["hparams"]["max_epochs"]):
        current_epoch_idx = epoch_idx
        logger.info(f"\nEpoch {epoch_idx + 1}/{config['hparams']['max_epochs']}")
        
        # --------------------
        # Training
        # --------------------
        model.train()
        _default_kwargs = dict(unit="GraphPairBatch", ncols=100)
        for batch_idx, batch in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"{'train':<5}",**_default_kwargs,
        ):
            optimizer.zero_grad()
            
            # Forward pass
            avg_loss, pred,true = feed_forward_step(
                model=model,
                batch=batch,
                loss_callables=loss_callables,
                is_train=True,
                edge_cutoff=config["hparams"]["edge_cutoff"],
                num_edge_cutoff=config["hparams"]["num_edge_cutoff"],
            )
            train_pred.append(pred)
            train_true.append(true)
            
            if tb_writer is not None:
                global_step = epoch_idx * len(train_loader) + batch_idx
                for key, value in step_metrics.items():
                    tb_writer.add_scalar(key, value, global_step)
            
            
            # Backward pass and optimize
            avg_loss.backward()
            optimizer.step()

        # Calculate epoch-level training metrics
        # avg_epoch_edge_metrics, avg_epoch_epi_metrics = epoch_end(train_pred,train_true)
        train_epoch_metrics = {
            "trainEpoch/loss": avg_loss.item(),
            # "trainEpoch/edge_auprc": avg_epoch_edge_metrics["auprc"].item(),
            # "trainEpoch/edge_mcc": avg_epoch_edge_metrics["mcc"].item(),
            # "trainEpoch/epi_auprc": avg_epoch_epi_metrics["auprc"].item(),
            # "trainEpoch/epi_mcc": avg_epoch_epi_metrics["mcc"].item(),
        }
        logger.info(f"Training metrics:\n{pformat(train_epoch_metrics)}")
        train_pred,train_true =  [],[]
        
        # Log epoch metrics to TensorBoard
        if tb_writer is not None:
            for key, value in train_epoch_metrics.items():
                tb_writer.add_scalar(key, value, epoch_idx)
        

        # --------------------
        # Validation
        # --------------------
        val_pred,val_true = [],[]
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc=f"{'val':<5}",
                unit="GraphPairBatch",
                ncols=100,
            ):
                # Forward pass
                avg_loss, pred,true = feed_forward_step(
                    model=model,
                    batch=batch,
                    loss_callables=loss_callables,
                    is_train=False,
                    edge_cutoff=config["hparams"]["edge_cutoff"],
                    num_edge_cutoff=config["hparams"]["num_edge_cutoff"],
                )
                val_pred.append(pred)
                val_true.append(true)
                
                if tb_writer is not None:
                    global_step = epoch_idx * len(val_loader) + batch_idx
                    for key, value in step_metrics.items():
                        tb_writer.add_scalar(key, value, global_step)
                

        # Calculate epoch-level validation metrics
        avg_epoch_edge_metrics, avg_epoch_epi_metrics = epoch_end(val_pred,val_true)
        val_epoch_metrics = {
            "valEpoch/edge_auprc": avg_epoch_edge_metrics["auprc"].item(),
            "valEpoch/edge_mcc": avg_epoch_edge_metrics["mcc"].item(),
            "valEpoch/epi_auprc": avg_epoch_epi_metrics["auprc"].item(),
            "valEpoch/epi_mcc": avg_epoch_epi_metrics["mcc"].item(),
            "valEpoch/epi_thr": avg_epoch_epi_metrics["best_thr"].item()
        }
        logger.info(f"Validation metrics:\n{pformat(val_epoch_metrics)}")
        val_pred,val_true =  [],[]
        
        # Log epoch metrics to TensorBoard
        if tb_writer is not None:
            for key, value in val_epoch_metrics.items():
                tb_writer.add_scalar(key, value, epoch_idx)
        best_thr = avg_epoch_epi_metrics["best_thr"].item()
        # --------------------
        # Testing (per epoch)
        # --------------------
        test_pred,test_true = [],[]
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(test_loader),
                total=len(test_loader),
                desc=f"{'test':<5}",
                unit="GraphPairBatch",
                ncols=100,
            ):
                # Forward pass
                avg_loss, pred,true = feed_forward_step(
                    model=model,
                    batch=batch,
                    loss_callables=loss_callables,
                    is_train=False,
                    edge_cutoff=config["hparams"]["edge_cutoff"],
                    num_edge_cutoff=config["hparams"]["num_edge_cutoff"],
                )
                test_pred.append(pred)
                test_true.append(true)
                
                if tb_writer is not None:
                    global_step = epoch_idx * len(test_loader) + batch_idx
                    for key, value in step_metrics.items():
                        tb_writer.add_scalar(key, value, global_step)
                

        # Calculate epoch-level test metrics
        avg_epoch_edge_metrics, avg_epoch_epi_metrics = epoch_end(test_pred,test_true,fixed_threshold = None)
        test_epoch_metrics = {
            "testEpoch/edge_auprc": avg_epoch_edge_metrics["auprc"].item(),
            "testEpoch/edge_mcc": avg_epoch_edge_metrics["mcc"].item(),
            "testEpoch/epi_auprc": avg_epoch_epi_metrics["auprc"].item(),
            "testEpoch/epi_auroc": avg_epoch_epi_metrics["auroc"].item(),
            "testEpoch/epi_mcc": avg_epoch_epi_metrics["mcc"].item(),
            "testEpoch/epi_precision": avg_epoch_epi_metrics["precision"].item(),
            "testEpoch/epi_recall": avg_epoch_epi_metrics["recall"].item(),
            "testEpoch/epi_f1": avg_epoch_epi_metrics["f1"].item(),
            "best_thr": avg_epoch_epi_metrics["best_thr"],
        }
        logger.info(f"Test metrics:\n{pformat(test_epoch_metrics)}")
        test_pred,test_true =  [],[]
        
        # Log epoch metrics to TensorBoard
        if tb_writer is not None:
            for key, value in test_epoch_metrics.items():
                tb_writer.add_scalar(key, value, epoch_idx)
        

        # --------------------
        # Callbacks
        # --------------------
        # Model checkpoint
        if ck_model_ckpt is not None:
            ck_model_ckpt.step(
                metrics=val_epoch_metrics,
                model=model,
                epoch=epoch_idx,
                optimizer=optimizer,
            )
        if ck_model_ckpt_edge is not None:
            ck_model_ckpt_edge.step(
                metrics=val_epoch_metrics,
                model=model,
                epoch=epoch_idx,
                optimizer=optimizer,
            )
        
        # Early stopping
        if (ck_early_stop is not None) and ck_early_stop.early_stop(epoch=epoch_idx, metrics=val_epoch_metrics):
            logger.info(f"Early stopping triggered at epoch {epoch_idx + 1}")
            break
        
        # Learning rate scheduler
        exec_lr_scheduler(ck_lr_scheduler, config, val_epoch_metrics=val_epoch_metrics)
        for param_group in optimizer.param_groups:
            logger.info(f"Learning rate after epoch {epoch_idx + 1}: {param_group['lr']:.6f}")

    # --------------------
    # Final Model Saving
    # --------------------
    if ck_model_ckpt is not None:
        # Save last model
        metric_name = config["callbacks"]["model_checkpoint"]["metric_name"]
        ck_model_ckpt.save_last(
            epoch=current_epoch_idx,
            model=model,
            optimizer=optimizer,
            metric_value=val_epoch_metrics[metric_name],
            upload=False,  # No wandb, disable upload
        )
        
        # Save best k models
        ck_model_ckpt.save_best_k(keep_interim=config["keep_interim_ckpts"])
        logger.info(f"Saved best {ck_model_ckpt.k} models based on {metric_name}")

    if ck_model_ckpt_edge is not None:
        ck_model_ckpt_edge.save_best_k(keep_interim=config["keep_interim_ckpts"])
        edge_metric_name = config["callbacks"]["model_checkpoint_edge"]["metric_name"]
        logger.info(f"Saved best {ck_model_ckpt_edge.k} edge models based on {edge_metric_name}")

    # --------------------
    # Final Testing with Best Model
    # --------------------
    final_pred,final_true = [],[]
    if ck_model_ckpt is not None:
        # Load best model
        ckpt_data = ck_model_ckpt.load_best()
        model.load_state_dict(ckpt_data["model_state_dict"])
        logger.info("Loaded best model for final testing")

        # Final test
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(test_loader),
                total=len(test_loader),
                desc=f"{'testF':<5}",
                unit="graph",
                ncols=100,
            ):
                avg_loss, pred,true = feed_forward_step(
                    model=model,
                    batch=batch,
                    loss_callables=loss_callables,
                    is_train=False,
                    edge_cutoff=config["hparams"]["edge_cutoff"],
                    num_edge_cutoff=config["hparams"]["num_edge_cutoff"],
                )
                final_pred.append(pred)
                final_true.append(true)

        # Calculate final test metrics
        avg_epoch_edge_metrics, avg_epoch_epi_metrics = epoch_end(final_pred,final_true)
        final_test_metrics = {
            "testFinal/edge_auprc": avg_epoch_edge_metrics["auprc"].item(),
            "testFinal/edge_mcc": avg_epoch_edge_metrics["mcc"].item(),
            "testFinal/epi_auprc": avg_epoch_epi_metrics["auprc"].item(),
            "testFinal/epi_aueoc": avg_epoch_epi_metrics["auroc"].item(),
            "testFinal/epi_mcc": avg_epoch_epi_metrics["mcc"].item(),
            "testFinal/epi_recall": avg_epoch_epi_metrics["recall"].item(),
            "testFinal/epi_precision": avg_epoch_epi_metrics["precision"].item(),
            "testFinal/epi_f1": avg_epoch_epi_metrics["f1"].item(),
        }
        logger.info(f"Final test metrics with best model:\n{pformat(final_test_metrics)}")
        
        # Log final metrics to TensorBoard
        if tb_writer is not None:
            for key, value in final_test_metrics.items():
                tb_writer.add_scalar(key, value, current_epoch_idx)

    # Close TensorBoard writer if used
    if tb_writer is not None:
        tb_writer.close()
        logger.info("TensorBoard writer closed")
