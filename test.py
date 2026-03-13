import os
import csv
import pickle
import torch
from models import *
from metrics import *
from utils import *
from config import DefaultConfig
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
configs = DefaultConfig()


def test(model, test_graphs,i,mx_roc,mx_pr):
    model.eval()
    all_trues = []
    all_preds = []
    print(mx_roc,mx_pr)

    mx = 0
    comb = []
    lab = []
    for test_g in test_graphs:
        if torch.cuda.is_available():
            test_ag_vertex = torch.FloatTensor(test_g['l_vertex']).cuda()
            test_ag_edge = torch.FloatTensor(test_g['l_edge']).cuda()
            test_ag_nh_indices = torch.LongTensor(test_g['l_hood_indices']).cuda()
            test_ag_label = torch.LongTensor(test_g['label'])
            test_ag_indices = [index for index, _ in enumerate(test_g['label'])]
            test_ag_indices = torch.LongTensor(test_ag_indices).cuda()

        test_ag_label = (test_ag_label == 1).int()  # 转换 -1 为 0
        g_preds,u,v = model(test_ag_vertex, test_ag_edge, test_ag_nh_indices, test_ag_indices)
        g_preds = g_preds.data.cpu().numpy()
     
        test_ag_label = test_ag_label.numpy()
        all_preds.append(g_preds)
        all_trues.append(test_ag_label[:,1])


    all_trues = np.concatenate(all_trues, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    print(sum(all_trues))
    print(len(all_preds))
    
    auc_roc = compute_auc_roc(all_trues, all_preds)
    auc_pr = compute_auc_pr(all_trues, all_preds)
    
    if(auc_roc > mx_roc):
        mx_roc = auc_roc
    if(auc_pr > mx_pr):
        mx_pr = auc_pr
    all_preds = torch.tensor(all_preds, dtype=torch.float32).to(device)
    all_preds = torch.sigmoid(all_preds)
    all_preds = all_preds.cpu().detach().numpy()
    f_max, p_max, r_max, t_max, predictions_max = compute_performance(all_trues, all_preds)
    acc = compute_acc(all_trues, predictions_max)
    recall = compute_recall(all_trues,predictions_max)
    precision = compute_precision(all_trues,predictions_max)
    ba = compute_balanced_accuracy(all_trues,predictions_max)
    mcc = compute_mcc(all_trues, predictions_max)
    cm = confusion_matrix(all_trues,predictions_max)
    print(cm)
    print(acc, recall,ba,auc_roc, auc_pr, mcc, f_max, p_max, r_max, t_max,precision)
    return [acc, recall, ba, auc_roc, auc_pr, mcc, f_max, p_max, r_max, t_max,mx_roc,mx_pr]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    test_model = BiLSTMNodeEdgeAverageModel()

    if torch.cuda.is_available():
        model = test_model.cuda()

    test_path = configs.test_dataset_path

    with open(test_path, 'rb') as f:
        test_data = pickle.load(f,encoding='latin1')

    from utils import propress_data
    propress_data(test_data)

    with open(os.path.join(configs.save_path, 'results.csv'), 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['acc', 'recall','ba','auc_roc', 'auc_pr', 'mcc', 'f_max', 'p_max', 'r_max', 't_max'])
    mx_pr,mx_roc = 0,0
    for i in range(10):
        print('experiment:', i + 1)
        test_model_path = os.path.join(configs.save_path, 'model{}.tar'.format(i + 1))

        test_model_sd = torch.load(test_model_path)
        test_model.load_state_dict(test_model_sd)
        if torch.cuda.is_available():
            model = test_model.cuda()

        experiment_results = test(test_model, test_data,i,mx_roc,mx_pr)
        [mx_roc,mx_pr] = experiment_results[10: 12]
    
        with open(os.path.join(configs.save_path, 'results.csv'), 'a+', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(experiment_results)
