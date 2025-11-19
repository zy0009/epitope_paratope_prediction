"""
credit: https://stackoverflow.com/a/73704579
Example usage:
early_stopper = EarlyStopper(patience=3, min_delta=10)
for epoch in np.arange(n_epochs):
    train_loss = train_one_epoch(model, train_loader)
    validation_loss = validate_one_epoch(model, validation_loader)
    if early_stopper.early_stop(validation_loss):
        break
"""
import os
import shutil
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Union
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import wandb

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s {%(pathname)s:%(lineno)d} [%(levelname)s] %(name)s - %(message)s [%(threadName)s]',
                    datefmt='%H:%M:%S')


class EarlyStopper:
    def __init__(
        self,
        patience=1,
        min_delta=0,
        minimize: bool = True,
        metric_name: str="val_loss"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.minimize = minimize
        self.best_val = np.inf if minimize else -np.inf
        self.metric_name = metric_name

    def early_stop(self, epoch: int, metrics: Dict):  # sourcery skip: merge-else-if-into-elif
        assert self.metric_name in metrics.keys(), f"provided metric_name {self.metric_name} not in metrics.\nValid keys are {metrics.keys()}"
        value = metrics[self.metric_name]
        # minimize
        if self.minimize:
            if value < self.best_val:
                self.reset_counter(value, epoch)
            elif value > (self.best_val + self.min_delta):
                self.counter += 1
                logging.info(f"Epoch {epoch}, EarlyStopper counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    return True
        else:
            if value > self.best_val:
                self.reset_counter(value, epoch)
            elif value <= (self.best_val - self.min_delta):
                self.counter += 1
                logging.info(f"Epoch {epoch}, EarlyStopper counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    return True
        return False

    def reset_counter(self, value: Tensor, epoch: int):
        """ reset counter and best_val

        Args:
            value (Tensor): metric value to be compared
            epoch (int): epoch number
        """
        self.best_val = value
        self.counter = 0
        logging.info(f"Epoch {epoch}, EarlyStopper reset counter")


class ModelCheckpoint:
    def __init__(
        self,
        save_dir: Union[Path, str],
        k: int = 1,
        minimize: bool = True,
        metric_name: str = "val_loss",
    ):
        self.save_dir = Path(save_dir)
        self.k = k
        self.minimize = minimize
        self.metric_name = metric_name

        # 统一用 float 初始化 best_k_metric_value
        self.best_k_metric_value: List[float] = [float(np.inf)] * k
        if not self.minimize:
            self.best_k_metric_value = [float(-np.inf)] * k

        self.best_k_epoch: List[int] = [-1] * k
        self.best_k_fp: List[Optional[Path]] = [None] * k
        """
        best_k_metric_value: 指标值（float），索引对应 epoch 和文件路径
        best_k_epoch       : 保存该指标对应的 epoch
        best_k_fp          : 保存该指标对应的 ckpt 路径
        """

        # create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # create interim directory to save state_dict files
        self.interim_dir = self.save_dir.joinpath("interim")
        self.interim_dir.mkdir(parents=True, exist_ok=True)

        # create best_k directory to save best k state_dict files
        self.best_k_dir = self.save_dir.joinpath(f"best_{k}")
        self.best_k_dir.mkdir(parents=True, exist_ok=True)

    def time_stamp(self):
        """ generate a time stamp, e.g. 20230611-204118 """
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def save_model(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metric_value: float,
    ) -> Path:
        """ save a model to interim directory """

        obj_to_save = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric_value": float(metric_value),
        }
        ckpt_path = self.interim_dir.joinpath(
            f"epoch{epoch}-{self.time_stamp()}.pt"
        )
        torch.save(obj_to_save, ckpt_path)
        return ckpt_path

    def update_best_k(self, epoch: int, metric_value: float, ckpt_path: Path):
        """ Update the best k metric value, epoch number, and file path """

        # 找到当前 best_k 里最差的那个（minimize: 最大的值；maximize: 最小的值）
        if self.minimize:
            worst_value = max(self.best_k_metric_value)
        else:
            worst_value = min(self.best_k_metric_value)

        idx = self.best_k_metric_value.index(worst_value)

        # 用新的更优模型覆盖
        self.best_k_metric_value[idx] = float(metric_value)
        self.best_k_epoch[idx] = epoch
        self.best_k_fp[idx] = ckpt_path

    def step(
        self,
        metrics: Dict,                    # dict of metrics
        epoch: int,                       # current epoch
        model: nn.Module,                 # model
        optimizer: torch.optim.Optimizer, # optimizer
    ):
        """
        每个 epoch 调用一次，根据 metrics[self.metric_name] 决定是否保存 / 更新 best_k
        """
        assert self.metric_name in metrics.keys(), \
            f"provided metric_name {self.metric_name} not in metrics.\nValid keys are {metrics.keys()}"

        # 统一把当前指标转为 float
        v = metrics[self.metric_name]
        if isinstance(v, torch.Tensor):
            v = float(v.detach().cpu())
        else:
            v = float(v)

        # 和当前 best_k 中的最差值比较
        if self.minimize:
            cur_worst = max(self.best_k_metric_value)
            is_better = v < cur_worst
        else:
            cur_worst = min(self.best_k_metric_value)
            is_better = v > cur_worst

        if is_better:
            ckpt_path = self.save_model(
                epoch=epoch, model=model, optimizer=optimizer, metric_value=v
            )
            self.update_best_k(epoch=epoch, metric_value=v, ckpt_path=ckpt_path)

    def sort_best_k(self):
        """
        按指标排序，返回索引
        - minimize: 升序（最小的在前）
        - maximize: 降序（最大的在前）
        """
        # 这里 best_k_metric_value 已经是纯 float 列表
        values = torch.tensor(self.best_k_metric_value, dtype=torch.float32)
        indices = torch.argsort(values)  # ascending

        return indices if self.minimize else torch.flip(indices, dims=(0,))

    def save_best_k(
        self,
        keep_interim: bool = True,
    ):
        """
        Save the best k models and create soft links rank_0.pt, rank_1.pt, ...
        """
        # sort best k
        indices = self.sort_best_k()  # the best at index 0

        # save the best k models to self.best_k_dir
        for i, j in enumerate(indices):
            epoch = self.best_k_epoch[j]
            interim_ckpt_path = self.best_k_fp[j]
            if interim_ckpt_path is None:
                continue
            shutil.copy(
                interim_ckpt_path,
                self.best_k_dir.joinpath(f"rank_{i}-epoch_{epoch}.pt"),
            )

        # create a soft link to the best k models
        for i in range(self.k):
            dst = self.save_dir.joinpath(f"rank_{i}.pt")
            if dst.exists():
                os.remove(dst)
                logging.warn(f"soft link {dst} already exists. It is removed.")
            candidates = list(self.best_k_dir.glob(f"rank_{i}*.pt"))
            if len(candidates) == 0:
                continue
            os.symlink(
                src=os.path.relpath(candidates[0], self.save_dir),
                dst=dst,
            )

        # remove the interim directory if keep_interim is False
        if not keep_interim:
            shutil.rmtree(self.interim_dir)

    def save_last(
        self,
        *args,
        upload: bool = True,
        wandb_run: "wandb.sdk.wandb_run.Run" = None,
        **kwargs,
    ):
        """
        Wrapper to save the last model
        args and kwargs are passed to self.save_model
        """
        ckpt_path = self.save_model(*args, **kwargs)
        shutil.copy(
            ckpt_path,
            self.save_dir
        )
        # if the soft link already exists, remove it
        last_link = self.save_dir.joinpath("last.pt")
        if last_link.exists():
            os.remove(last_link)
            logging.warn(f"soft link {last_link} already exists. It is removed.")
        # create a soft link to the last model
        os.symlink(
            src=os.path.relpath(self.save_dir.joinpath(ckpt_path.name), self.save_dir),
            dst=last_link,
        )

        # 你现在没用 wandb，可以保持 upload=False 或直接忽略这部分
        if upload and wandb_run is not None:
            artifact = wandb.Artifact(
                name="last_epoch_checkpoint",
                type="model",
                metadata=dict(
                    metric_name=self.metric_name,
                ),
            )
            artifact.add_file(ckpt_path)
            wandb_run.log_artifact(artifact)

    def load_best(self) -> Dict[str, Any]:
        """
        Load the best model from the best_k_dir
        CAUTION: this should only be called when training is done
            i.e. after self.save_best_k() is called
        """
        return torch.load(
            self.save_dir.joinpath("rank_0.pt")
        )

    def upload_best_k_to_wandb(self, wandb_run: "wandb.sdk.wandb_run.Run", suffix: str = None):
        """
        Upload the best k models to wandb as artifacts
        CAUTION: only call this after training is done, self.save_best_k() must be called
        """
        suffix = suffix or ""
        artifact = wandb.Artifact(
            name="best_k_models" + suffix,
            type="model",
            metadata=dict(
                metric_name=self.metric_name,
            ),
        )
        for i in range(self.k):
            real_path = Path(os.path.realpath(self.save_dir.joinpath(f"rank_{i}.pt")))
            artifact.add_file(real_path)
        wandb_run.log_artifact(artifact)
