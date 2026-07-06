import os
import csv
import pickle
import numpy as np
import torch
from models import *
from metrics import *
from utils import *
from config import DefaultConfig
from sklearn.metrics import confusion_matrix
configs = DefaultConfig()

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


def test(model, test_graphs, device, i, mx_roc, mx_pr):
    model.eval()
    all_trues = []
    all_preds = []

    mx = 0
    with torch.no_grad():
        for test_g in test_graphs:
            test_ag_vertex, test_ag_edge, test_ag_nh_indices, test_ag_label, test_ag_indices = graph_to_tensors(test_g, device)
            g_preds, u, v = model(test_ag_vertex, test_ag_edge, test_ag_nh_indices, test_ag_indices)
            all_preds.append(g_preds.detach().cpu().numpy())
            all_trues.append(test_ag_label[:, 1].detach().cpu().numpy())


    all_trues = np.concatenate(all_trues, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    all_probs = sigmoid_np(all_preds)
    auc_roc = compute_auc_roc(all_trues, all_probs)
    auc_pr = compute_auc_pr(all_trues, all_probs)
    
    if(auc_roc > mx_roc):
        mx_roc = auc_roc
    if(auc_pr > mx_pr):
        mx_pr = auc_pr
    f_max, p_max, r_max, t_max, predictions_max = compute_performance(all_trues, all_probs)
    acc = compute_acc(all_trues, predictions_max)
    recall = compute_recall(all_trues,predictions_max)
    precision = compute_precision(all_trues,predictions_max)
    ba = compute_balanced_accuracy(all_trues,predictions_max)
    mcc = compute_mcc(all_trues, predictions_max)
    cm = confusion_matrix(all_trues, predictions_max, labels=[0, 1])
    print(cm)
    print(acc, recall,ba,auc_roc, auc_pr, mcc, f_max, p_max, r_max, t_max,precision)
    return [acc, recall, ba, auc_roc, auc_pr, mcc, f_max, p_max, r_max, t_max,mx_roc,mx_pr]


def summarize_results(results):
    arr = np.asarray(results, dtype=float)
    names = ['acc', 'recall', 'ba', 'auc_roc', 'auc_pr', 'mcc', 'f_max', 'p_max', 'r_max', 't_max']
    rows = []
    n = arr.shape[0]
    for idx, name in enumerate(names):
        values = arr[:, idx]
        mean = np.nanmean(values)
        std = np.nanstd(values, ddof=1) if n > 1 else 0.0
        ci95 = ci95_half_width(std, n)
        rows.append([name, mean, std, ci95])
    return rows


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    test_path = configs.test_dataset_path

    with open(test_path, 'rb') as f:
        test_data = pickle.load(f,encoding='latin1')

    from utils import propress_data
    propress_data(test_data)

    with open(os.path.join(configs.save_path, 'results.csv'), 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['seed', 'acc', 'recall','ba','auc_roc', 'auc_pr', 'mcc', 'f_max', 'p_max', 'r_max', 't_max', 'mx_roc', 'mx_pr'])
    mx_pr,mx_roc = 0,0
    all_results = []
    for i, seed in enumerate(configs.seeds):
        print('experiment:', i + 1)
        test_model = BiLSTMNodeEdgeAverageModel().to(device)
        test_model_path = os.path.join(configs.save_path, 'model{}.tar'.format(i + 1))
        if not os.path.exists(test_model_path):
            print('missing checkpoint:', test_model_path)
            continue

        test_model_sd = torch.load(test_model_path, map_location=device)
        try:
            test_model.load_state_dict(test_model_sd)
        except RuntimeError as exc:
            print('checkpoint is incompatible with the current architecture:', test_model_path)
            print(exc)
            continue

        experiment_results = test(test_model, test_data, device, i, mx_roc, mx_pr)
        [mx_roc,mx_pr] = experiment_results[10: 12]
        all_results.append(experiment_results[:10])
    
        with open(os.path.join(configs.save_path, 'results.csv'), 'a+', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([seed] + experiment_results)

    if all_results:
        with open(os.path.join(configs.save_path, 'results_summary.csv'), 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['metric', 'mean', 'std', 'ci95'])
            csv_writer.writerows(summarize_results(all_results))
