import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropy(object):
    def __init__(self, neg_wt, device):
        neg_wt = torch.FloatTensor([neg_wt, 1]).to(device)
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss(weight=neg_wt, reduction='sum')

    def compute_loss(self, pred, true):
        true = true[:, 1]  
        true = true.long().to(self.device)
        # get preds with 1 dim, turn to 2 dim
        pred_ = torch.cat([-pred, pred], axis=1).to(self.device)
        loss = self.loss_fn(pred_, true) / len(true)
        return loss

    def computer_loss(self, pred, true):
        return self.compute_loss(pred, true)


class CrossModalContrastiveLoss(object):
    def __init__(
        self,
        temperature=0.1,
        label_aware=True,
        negative_mode='in_batch',
        num_prototypes=8,
        prototype_iters=5,
        device=None,
    ):
        self.temperature = temperature
        self.label_aware = label_aware
        self.negative_mode = negative_mode
        self.num_prototypes = max(1, int(num_prototypes))
        self.prototype_iters = max(1, int(prototype_iters))
        self.device = device

    def compute_loss(
        self,
        global_feat,
        local_feat,
        true=None,
        protein_ids=None,
        negative_global_feat=None,
        negative_local_feat=None,
        negative_labels=None,
        negative_protein_ids=None,
    ):
        global_feat = F.normalize(global_feat, dim=1)
        local_feat = F.normalize(local_feat, dim=1)

        if self.negative_mode == 'cross_protein':
            return self._compute_cross_protein_loss(
                global_feat,
                local_feat,
                true,
                protein_ids,
                negative_global_feat,
                negative_local_feat,
                negative_labels,
                negative_protein_ids,
            )
        if self.negative_mode == 'cluster':
            return self._compute_cluster_loss(
                global_feat,
                local_feat,
                negative_global_feat,
                negative_local_feat,
            )
        if self.negative_mode != 'in_batch':
            raise ValueError("negative_mode must be 'in_batch', 'cross_protein', or 'cluster'")

        logits = torch.matmul(global_feat, local_feat.t()) / self.temperature
        targets = torch.arange(logits.size(0), dtype=torch.long, device=logits.device)

        if self.label_aware and true is not None and logits.size(0) > 1:
            labels = true[:, 1].long().to(logits.device)
            same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
            false_negative_mask = same_label & ~torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(false_negative_mask, -1e9)

        loss_global_to_local = F.cross_entropy(logits, targets)
        loss_local_to_global = F.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_global_to_local + loss_local_to_global)

    def _compute_cross_protein_loss(
        self,
        global_feat,
        local_feat,
        true,
        protein_ids,
        negative_global_feat,
        negative_local_feat,
        negative_labels,
        negative_protein_ids,
    ):
        if (
            negative_global_feat is None
            or negative_local_feat is None
            or negative_labels is None
            or negative_protein_ids is None
            or negative_global_feat.size(0) == 0
        ):
            return global_feat.new_tensor(0.0)

        negative_global_feat = F.normalize(negative_global_feat.to(global_feat.device), dim=1)
        negative_local_feat = F.normalize(negative_local_feat.to(local_feat.device), dim=1)
        negative_labels = negative_labels.to(global_feat.device)
        negative_protein_ids = negative_protein_ids.to(global_feat.device)
        protein_ids = protein_ids.to(global_feat.device)

        labels = None
        if true is not None:
            labels = true[:, 1].long().to(global_feat.device)

        loss_global_to_local = self._cross_protein_direction_loss(
            global_feat,
            local_feat,
            negative_local_feat,
            labels,
            protein_ids,
            negative_labels,
            negative_protein_ids,
        )
        loss_local_to_global = self._cross_protein_direction_loss(
            local_feat,
            global_feat,
            negative_global_feat,
            labels,
            protein_ids,
            negative_labels,
            negative_protein_ids,
        )
        return 0.5 * (loss_global_to_local + loss_local_to_global)

    def _cross_protein_direction_loss(
        self,
        anchor_feat,
        positive_feat,
        negative_feat,
        anchor_labels,
        anchor_protein_ids,
        negative_labels,
        negative_protein_ids,
    ):
        positive_logits = (anchor_feat * positive_feat).sum(dim=1, keepdim=True) / self.temperature
        negative_logits = torch.matmul(anchor_feat, negative_feat.t()) / self.temperature

        same_protein = anchor_protein_ids.unsqueeze(1) == negative_protein_ids.unsqueeze(0)
        invalid_negatives = same_protein
        if self.label_aware and anchor_labels is not None:
            same_label = anchor_labels.unsqueeze(1) == negative_labels.unsqueeze(0)
            invalid_negatives = invalid_negatives | same_label
        negative_logits = negative_logits.masked_fill(invalid_negatives, -1e9)

        logits = torch.cat([positive_logits, negative_logits], dim=1)
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, targets)

    def _compute_cluster_loss(
        self,
        global_feat,
        local_feat,
        negative_global_feat,
        negative_local_feat,
    ):
        global_candidates = self._cluster_candidates(global_feat, negative_global_feat)
        local_candidates = self._cluster_candidates(local_feat, negative_local_feat)

        loss_global_to_local = self._cluster_direction_loss(global_feat, local_candidates)
        loss_local_to_global = self._cluster_direction_loss(local_feat, global_candidates)
        return 0.5 * (loss_global_to_local + loss_local_to_global)

    def _cluster_candidates(self, current_feat, memory_feat):
        if memory_feat is None or memory_feat.size(0) == 0:
            return current_feat
        memory_feat = F.normalize(memory_feat.to(current_feat.device), dim=1)
        return torch.cat([current_feat, memory_feat], dim=0)

    def _cluster_direction_loss(self, anchor_feat, candidate_feat):
        if candidate_feat.size(0) == 0:
            return anchor_feat.sum() * 0.0

        cluster_feat = torch.cat([anchor_feat, candidate_feat], dim=0)
        centers = self._cluster_prototypes(cluster_feat)
        anchor_assignments = self._assign_clusters(anchor_feat, centers)
        candidate_assignments = self._assign_clusters(candidate_feat, centers)

        logits = torch.matmul(anchor_feat, candidate_feat.t()) / self.temperature
        positive_mask = anchor_assignments.unsqueeze(1) == candidate_assignments.unsqueeze(0)
        valid_anchor_mask = positive_mask.any(dim=1)
        if not valid_anchor_mask.any():
            return anchor_feat.sum() * 0.0

        logits = logits[valid_anchor_mask]
        positive_mask = positive_mask[valid_anchor_mask]
        log_probs = F.log_softmax(logits, dim=1)
        positive_log_probs = log_probs.masked_fill(~positive_mask, 0.0).sum(dim=1)
        positive_counts = positive_mask.sum(dim=1).clamp_min(1)
        return -(positive_log_probs / positive_counts).mean()

    def _cluster_prototypes(self, features):
        features = F.normalize(features, dim=1)
        k = min(self.num_prototypes, features.size(0))
        if features.size(0) == k:
            return features

        init_indices = torch.linspace(0, features.size(0) - 1, steps=k, device=features.device).long()
        centers = features.index_select(0, init_indices)
        for _ in range(self.prototype_iters):
            distances = ((features.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(dim=2)
            assignments = distances.argmin(dim=1)
            new_centers = []
            for cluster_idx in range(k):
                cluster_points = features[assignments == cluster_idx]
                if cluster_points.size(0) == 0:
                    new_centers.append(centers[cluster_idx])
                else:
                    new_centers.append(cluster_points.mean(dim=0))
            centers = F.normalize(torch.stack(new_centers, dim=0), dim=1)
        return centers

    def _assign_clusters(self, features, centers):
        distances = ((features.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(dim=2)
        return distances.argmin(dim=1)
