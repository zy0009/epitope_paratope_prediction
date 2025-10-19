import torch
import torch.nn as nn


class WeightedCrossEntropy(object):
    def __init__(self, neg_wt, device):
        neg_wt = torch.FloatTensor([neg_wt, 1]).to(device)
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss(weight=neg_wt, reduction='sum')

    def computer_loss(self, pred, true):
        # 假设 true 的第二维是标签，提取出来
        true = true[:, 1]  # 提取第二维作为标签
        # 确保 true 是 torch.long 类型
        true = true.long().to(self.device)
        # get preds with 1 dim, turn to 2 dim
        pred_ = torch.cat([-pred, pred], axis=1).to(self.device)
        loss = self.loss_fn(pred_, true) / len(true)
        return loss


