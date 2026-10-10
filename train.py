import os
import csv
import math
import numpy as np
import random
import torch
import pickle
from utils import *
from models import *
from metrics import *
from losses import WeightedCrossEntropy, CrossModalContrastiveLoss
from config import DefaultConfig
configs = DefaultConfig()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

T_CRITICAL_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def ci95_half_width(std, n):
    if n <= 1:
        return 0.0
    t_critical = T_CRITICAL_975.get(n - 1, 1.96)
    return t_critical * std / np.sqrt(n)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def graph_to_tensors(graph, device):
    vertex = torch.as_tensor(graph['l_vertex'], dtype=torch.float32, device=device)
    edge = torch.as_tensor(graph['l_edge'], dtype=torch.float32, device=device)
    nh_indices = torch.as_tensor(graph['l_hood_indices'], dtype=torch.long, device=device)
    labels = torch.as_tensor(graph['label'], dtype=torch.long, device=device).clone()
    labels[:, 1] = (labels[:, 1] == 1).long()
    indices = torch.arange(labels.size(0), dtype=torch.long, device=device)
    return vertex, edge, nh_indices, labels, indices


def sigmoid_np(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def contrastive_weight(epoch_index, total_epochs):
    if not getattr(configs, 'use_contrastive', False):
        return 0.0
    if total_epochs <= 1:
        progress = 1.0
    else:
        progress = epoch_index / float(total_epochs - 1)
    return configs.contrastive_initial_weight * math.exp(-configs.contrastive_decay_rate * progress)


def evaluate(model, graphs, device, threshold=None):
    model.eval()
    all_trues = []
    all_preds = []
    with torch.no_grad():
        for graph in graphs:
            vertex, edge, nh_indices, labels, indices = graph_to_tensors(graph, device)
            preds, _, _ = model(vertex, edge, nh_indices, indices)
            all_preds.append(preds.detach().cpu().numpy())
            all_trues.append(labels[:, 1].detach().cpu().numpy())

    all_trues = np.concatenate(all_trues, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = sigmoid_np(all_preds)

    auc_roc = compute_auc_roc(all_trues, all_probs)
    auc_pr = compute_auc_pr(all_trues, all_probs)
    f_max, p_max, r_max, t_max, predictions_max = compute_performance(
        all_trues, all_probs, threshold=threshold
    )
    return {
        'acc': compute_acc(all_trues, predictions_max),
        'recall': compute_recall(all_trues, predictions_max),
        'ba': compute_balanced_accuracy(all_trues, predictions_max),
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'mcc': compute_mcc(all_trues, predictions_max),
        'f_max': f_max,
        'p_max': p_max,
        'r_max': r_max,
        't_max': t_max,
    }


def summarize_results(results):
    metric_names = ['acc', 'recall', 'ba', 'auc_roc', 'auc_pr', 'mcc', 'f_max', 'p_max', 'r_max', 't_max']
    rows = []
    n = len(results)
    for metric_name in metric_names:
        values = np.asarray([result[metric_name] for result in results], dtype=float)
        mean = np.nanmean(values)
        std = np.nanstd(values, ddof=1) if n > 1 else 0.0
        ci95 = ci95_half_width(std, n)
        rows.append([metric_name, mean, std, ci95])
    return rows


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def subset_graphs(graphs, indices):
    return [graphs[index] for index in indices]


def build_k_fold_indices(num_items, k_folds, seed):
    indices = list(range(num_items))
    rng = random.Random(seed)
    rng.shuffle(indices)
    fold_sizes = [num_items // k_folds] * k_folds
    for i in range(num_items % k_folds):
        fold_sizes[i] += 1

    folds = []
    start = 0
    for fold_size in fold_sizes:
        end = start + fold_size
        folds.append(indices[start:end])
        start = end
    return folds


def split_train_val_indices(indices, val_ratio, seed):
    shuffled = list(indices)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_size = max(1, int(round(len(shuffled) * val_ratio)))
    val_size = min(val_size, len(shuffled) - 1)
    val_indices = shuffled[:val_size]
    train_indices = shuffled[val_size:]
    return train_indices, val_indices


class ContrastiveMemoryBank(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.global_feat = None
        self.local_feat = None
        self.labels = None
        self.protein_ids = None

    def get(self, device):
        if self.global_feat is None:
            return None, None, None, None
        return (
            self.global_feat.to(device),
            self.local_feat.to(device),
            self.labels.to(device),
            self.protein_ids.to(device),
        )

    def add(self, global_feat, local_feat, labels, protein_ids):
        if self.max_size <= 0:
            return
        global_feat = global_feat.detach().cpu()
        local_feat = local_feat.detach().cpu()
        labels = labels[:, 1].detach().long().cpu()
        protein_ids = protein_ids.detach().long().cpu()

        if self.global_feat is None:
            self.global_feat = global_feat
            self.local_feat = local_feat
            self.labels = labels
            self.protein_ids = protein_ids
        else:
            self.global_feat = torch.cat([self.global_feat, global_feat], dim=0)
            self.local_feat = torch.cat([self.local_feat, local_feat], dim=0)
            self.labels = torch.cat([self.labels, labels], dim=0)
            self.protein_ids = torch.cat([self.protein_ids, protein_ids], dim=0)

        if self.global_feat.size(0) > self.max_size:
            self.global_feat = self.global_feat[-self.max_size:]
            self.local_feat = self.local_feat[-self.max_size:]
            self.labels = self.labels[-self.max_size:]
            self.protein_ids = self.protein_ids[-self.max_size:]


def write_results_header(path):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment', 'seed', 'fold', 'train_size', 'val_size', 'test_size', 'acc', 'recall', 'ba', 'auc_roc', 'auc_pr', 'mcc', 'f_max', 'p_max', 'r_max', 't_max'])


def append_result(path, experiment, seed, fold, train_size, val_size, test_size, result):
    metric_names = ['acc', 'recall', 'ba', 'auc_roc', 'auc_pr', 'mcc', 'f_max', 'p_max', 'r_max', 't_max']
    with open(path, 'a+', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([experiment, seed, fold, train_size, val_size, test_size] + [result[name] for name in metric_names])


def write_thresholds_header(path):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment', 'seed', 'fold', 'validation_f1_threshold'])


def append_threshold(path, experiment, seed, fold, threshold):
    with open(path, 'a+', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([experiment, seed, fold, threshold])


def write_summary(path, results):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'mean', 'std', 'ci95'])
        writer.writerows(summarize_results(results))


def write_grouped_summary(path, grouped_results):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group', 'metric', 'mean', 'std', 'ci95'])
        for group_name, results in grouped_results.items():
            for metric_name, mean, std, ci95 in summarize_results(results):
                writer.writerow([group_name, metric_name, mean, std, ci95])


def train(model, train_graphs, val_graphs, num=1, re_train_start=0, b_loss=999):
    epochs = configs.epochs - re_train_start
    print('current experiment epochs:', epochs)
    batch_size = configs.batch_size
    lr = configs.learning_rate
    weight_decay = configs.weight_decay
    neg_wt = configs.neg_wt

    model_save_path = configs.save_path
    print(model_save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',      
        factor=0.95,      
        patience=3,        
    )
    loss_fn = WeightedCrossEntropy(neg_wt=neg_wt, device=device)
    contrastive_negative_mode = getattr(configs, 'contrastive_negative_mode', 'in_batch')
    contrastive_loss_fn = CrossModalContrastiveLoss(
        temperature=configs.contrastive_temperature,
        label_aware=configs.label_aware_contrastive,
        negative_mode=contrastive_negative_mode,
        num_prototypes=getattr(configs, 'contrastive_num_prototypes', 8),
        prototype_iters=getattr(configs, 'contrastive_prototype_iters', 5),
        device=device,
    )

    current_lr = optimizer.param_groups[0]['lr']
    train_losses = []
    val_losses = []
    best_loss = b_loss
    count = 0
    for e in range(epochs):
        print("Running {} epoch".format(e + 1 + re_train_start))
        c_weight = contrastive_weight(e, epochs)
        if getattr(configs, 'use_contrastive', False):
            print("contrastive weight:", c_weight)
        model.train()
        e_loss = 0.
        train_contrastive_bank = ContrastiveMemoryBank(configs.contrastive_memory_size)
        for graph_idx, train_g in enumerate(train_graphs):
            train_ag_vertex, train_ag_edge, train_ag_nh_indices, train_ag_label, train_ag_indices = graph_to_tensors(train_g, device)
            train_ag_indices = train_ag_indices[torch.randperm(train_ag_indices.size(0), device=device)]
            label_size = len(train_ag_label)
            iters = math.ceil(label_size / batch_size)
            g_loss = 0.

            for it in range(iters):
                
                start = it * batch_size
                end = start + batch_size
                if end > label_size:
                    end = label_size
                optimizer.zero_grad()
                
                batch_pred, global_ctx, local_ctx = model(
                    train_ag_vertex, 
                    train_ag_edge, 
                    train_ag_nh_indices, 
                    train_ag_indices[start:end]
                )
  
                batch_labels = train_ag_label[train_ag_indices[start:end]]
                main_loss = loss_fn.compute_loss(batch_pred, batch_labels)
                total_loss = main_loss
                if getattr(configs, 'use_contrastive', False):
                    global_z, local_z = model.contrastive_embeddings(global_ctx, local_ctx)
                    if contrastive_negative_mode in ('cross_protein', 'cluster'):
                        batch_protein_ids = torch.full(
                            (batch_labels.size(0),),
                            graph_idx,
                            dtype=torch.long,
                            device=device,
                        )
                        neg_global_z, neg_local_z, neg_labels, neg_protein_ids = train_contrastive_bank.get(device)
                        contrast_loss = contrastive_loss_fn.compute_loss(
                            global_z,
                            local_z,
                            batch_labels,
                            protein_ids=batch_protein_ids,
                            negative_global_feat=neg_global_z,
                            negative_local_feat=neg_local_z,
                            negative_labels=neg_labels,
                            negative_protein_ids=neg_protein_ids,
                        )
                        train_contrastive_bank.add(global_z, local_z, batch_labels, batch_protein_ids)
                    else:
                        contrast_loss = contrastive_loss_fn.compute_loss(global_z, local_z, batch_labels)
                    total_loss = total_loss + c_weight * contrast_loss
                
                total_loss.backward()
                optimizer.step()
                
                b_loss = total_loss.item()
                g_loss += b_loss
            g_loss /= iters
            e_loss += g_loss

        e_loss /= len(train_graphs)
        train_losses.append(e_loss)
        with open(os.path.join(model_save_path, 'epi_train_losses{}.txt'.format(num)), 'a+') as f:
            f.write(str(e_loss) + '\n')

        model.eval()
        e_loss = 0.
        val_contrastive_bank = ContrastiveMemoryBank(configs.contrastive_memory_size)
        with torch.no_grad():
            for graph_idx, val_g in enumerate(val_graphs):
                val_ag_vertex, val_ag_edge, val_ag_nh_indices, val_ag_label, val_ag_indices = graph_to_tensors(val_g, device)
                label_size = len(val_ag_label)
                iters = math.ceil(label_size / batch_size)
                g_loss = 0.
                for it in range(iters):
                    start = it * batch_size
                    end = min(start + batch_size, label_size)

                    batch_pred, global_ctx, local_ctx = model(
                        val_ag_vertex,
                        val_ag_edge,
                        val_ag_nh_indices,
                        val_ag_indices[start:end]
                    )
                    batch_labels = val_ag_label[val_ag_indices[start:end]]
                    main_loss = loss_fn.compute_loss(batch_pred, batch_labels)
                    total_loss = main_loss
                    if getattr(configs, 'use_contrastive', False):
                        global_z, local_z = model.contrastive_embeddings(global_ctx, local_ctx)
                        if contrastive_negative_mode in ('cross_protein', 'cluster'):
                            batch_protein_ids = torch.full(
                                (batch_labels.size(0),),
                                graph_idx,
                                dtype=torch.long,
                                device=device,
                            )
                            neg_global_z, neg_local_z, neg_labels, neg_protein_ids = val_contrastive_bank.get(device)
                            contrast_loss = contrastive_loss_fn.compute_loss(
                                global_z,
                                local_z,
                                batch_labels,
                                protein_ids=batch_protein_ids,
                                negative_global_feat=neg_global_z,
                                negative_local_feat=neg_local_z,
                                negative_labels=neg_labels,
                                negative_protein_ids=neg_protein_ids,
                            )
                            val_contrastive_bank.add(global_z, local_z, batch_labels, batch_protein_ids)
                        else:
                            contrast_loss = contrastive_loss_fn.compute_loss(global_z, local_z, batch_labels)
                        total_loss = total_loss + c_weight * contrast_loss
                    b_loss = total_loss.item()
                    g_loss += b_loss
                g_loss /= iters
                e_loss += g_loss
        e_loss /= len(val_graphs)
        val_losses.append(e_loss)
        with open(os.path.join(model_save_path, 'epi_val_losses{}.txt'.format(num)), 'a+') as f:
            f.write(str(e_loss) + '\n')

        if best_loss > val_losses[-1]:
            count = 0
            torch.save(model.state_dict(), os.path.join(os.path.join(model_save_path, "model{}.tar".format(num))))
            best_loss = val_losses[-1]
            print("UPDATE\tEpoch {}: train loss {}\tval loss {}".format(e + 1 + re_train_start, train_losses[-1], val_losses[-1]))
        else:
            count += 1
            if configs.use_early_stop and count >= configs.early_stop_patience:
                print("EARLY STOP\tEpoch {}: best val loss {}".format(e + 1 + re_train_start, best_loss))
                break
        scheduler.step(val_losses[-1])  
        prev_lr = current_lr 
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != prev_lr:
            print(f'\nlr change: {prev_lr} -> {current_lr}')


if __name__ == '__main__':
    train_path = configs.train_dataset_path
    val_path = configs.val_dataset_path
    test_path = configs.test_dataset_path

    train_data = load_pickle(train_path)
    val_data = load_pickle(val_path)
    test_data = load_pickle(test_path)
    
    if not os.path.exists(configs.save_path):
        os.makedirs(configs.save_path)

    results_path = os.path.join(configs.save_path, 'results.csv')
    summary_path = os.path.join(configs.save_path, 'results_summary.csv')
    fold_summary_path = os.path.join(configs.save_path, 'results_by_fold_summary.csv')
    thresholds_path = os.path.join(configs.save_path, 'validation_thresholds.csv')
    write_results_header(results_path)
    write_thresholds_header(thresholds_path)

    current_experiment = 1
    all_results = []
    results_by_fold = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if configs.use_k_fold:
        cv_data = train_data + val_data
        if len(cv_data) < configs.k_folds:
            raise ValueError('k_folds cannot exceed the number of samples')
        if len(configs.seeds) < configs.k_folds:
            raise ValueError('k-fold training requires at least k_folds seeds')
        print(
            'running {}-fold cross validation on {} train/validation complexes; '
            'the original test set with {} complexes is held out'.format(
                configs.k_folds,
                len(cv_data),
                len(test_data),
            )
        )

        fold_split_seed = configs.seeds[0]
        folds = build_k_fold_indices(len(cv_data), configs.k_folds, fold_split_seed)
        fold_seeds = configs.seeds[:configs.k_folds]
        print('fold split seed:', fold_split_seed)
        print('fold training seeds:', fold_seeds)

        for fold_idx, val_indices in enumerate(folds, start=1):
            seed = fold_seeds[fold_idx - 1]
            experiment_id = 'seed{}_fold{}'.format(seed, fold_idx)
            print('experiment:', current_experiment)
            print('seed:', seed)
            print('fold:', fold_idx)
            set_seed(seed)

            train_indices = [
                index
                for current_fold_idx, fold_indices in enumerate(folds, start=1)
                if current_fold_idx != fold_idx
                for index in fold_indices
            ]

            fold_train_data = subset_graphs(cv_data, train_indices)
            fold_val_data = subset_graphs(cv_data, val_indices)

            train_model = BiLSTMNodeEdgeAverageModel()
            for loss_name in ['epi_train_losses{}.txt', 'epi_val_losses{}.txt']:
                loss_path = os.path.join(configs.save_path, loss_name.format(experiment_id))
                if os.path.exists(loss_path):
                    os.remove(loss_path)

            train(train_model, fold_train_data, fold_val_data, experiment_id)

            checkpoint_path = os.path.join(configs.save_path, 'model{}.tar'.format(experiment_id))
            eval_model = BiLSTMNodeEdgeAverageModel().to(device)
            eval_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            validation_result = evaluate(eval_model, fold_val_data, device)
            validation_threshold = validation_result['t_max']
            append_threshold(
                thresholds_path,
                experiment_id,
                seed,
                fold_idx,
                validation_threshold,
            )
            print('validation-selected threshold:', validation_threshold)
            result = evaluate(eval_model, test_data, device, threshold=validation_threshold)
            print('test result:', result)
            append_result(
                results_path,
                experiment_id,
                seed,
                fold_idx,
                len(fold_train_data),
                len(fold_val_data),
                len(test_data),
                result,
            )
            all_results.append(result)
            results_by_fold.setdefault('fold{}'.format(fold_idx), []).append(result)
            current_experiment += 1
    else:
        for seed in configs.seeds:
            set_seed(seed)
            print('experiment:', current_experiment)
            print('seed:', seed)

            train_model = BiLSTMNodeEdgeAverageModel()
            for loss_name in ['epi_train_losses{}.txt', 'epi_val_losses{}.txt']:
                loss_path = os.path.join(configs.save_path, loss_name.format(current_experiment))
                if os.path.exists(loss_path):
                    os.remove(loss_path)

            train(train_model, train_data, val_data, current_experiment)

            checkpoint_path = os.path.join(configs.save_path, 'model{}.tar'.format(current_experiment))
            eval_model = BiLSTMNodeEdgeAverageModel().to(device)
            eval_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            validation_result = evaluate(eval_model, val_data, device)
            validation_threshold = validation_result['t_max']
            append_threshold(
                thresholds_path,
                current_experiment,
                seed,
                '',
                validation_threshold,
            )
            print('validation-selected threshold:', validation_threshold)
            result = evaluate(eval_model, test_data, device, threshold=validation_threshold)
            print('test result:', result)
            append_result(
                results_path,
                current_experiment,
                seed,
                '',
                len(train_data),
                len(val_data),
                len(test_data),
                result,
            )
            all_results.append(result)

            current_experiment += 1

    if all_results:
        write_summary(summary_path, all_results)
        if results_by_fold:
            write_grouped_summary(fold_summary_path, results_by_fold)
        print('saved per-seed results to:', results_path)
        print('saved summary results to:', summary_path)
        if results_by_fold:
            print('saved fold summary results to:', fold_summary_path)
