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

def tsne_visualization(embeddings, labels, title,sc, figsize=(10, 8), 
                       palette="viridis", save_path=None):
    """
    对嵌入向量进行t-SNE降维并可视化
    
    参数:
        embeddings: 输入的嵌入向量，形状为(n_samples, n_features)
        labels: 每个样本的标签，用于着色，形状为(n_samples,)
        title: 图表标题
        figsize: 图表大小
        palette: 颜色方案
        save_path: 保存图片的路径，为None则不保存
    """
    # 检查输入
    if isinstance(embeddings, torch.Tensor):
        # 如果是CUDA张量，先转移到CPU并转换为NumPy数组
        embeddings = embeddings.cpu().detach().numpy()
    
    # 检查标签是否为张量并处理
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().detach().numpy()
    if len(embeddings) != len(labels):
        raise ValueError("嵌入向量数量与标签数量不匹配")
    
    # 初始化t-SNE模型
    tsne = TSNE(
        n_components=2,        # 降维到2D以便可视化
        perplexity=30,         # 困惑度，通常在5-50之间
        learning_rate=200,     # 学习率，通常在10-1000之间
        n_iter=1000,           # 迭代次数
        random_state=42,       # 随机种子，保证结果可复现
        init='pca'             # 使用PCA初始化，更稳定
    )
    
    # 执行t-SNE降维
    print(f"正在对{len(embeddings)}个样本进行t-SNE降维...")
    embeddings_2d = tsne.fit_transform(embeddings)
    print("t-SNE降维完成")
    
    # 创建DataFrame用于绘图
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels
    })
    
    # 绘图
    plt.figure(figsize=figsize)
    sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='label',
        palette=palette,
        s=40,                  # 点的大小
        edgecolor='none',      # 去除边缘线
        linewidth=0            # 边缘线宽设为0
    )
    if sc is not None:
        plt.text(
            x=0.02,  # x坐标（相对值，0-1）
            y=0.02,  # y坐标（相对值，0-1）
            s=f'SC = {sc:.4f}',  # 显示文本
            fontsize=12,
            fontweight='bold',
            ha='left',  # 水平对齐：对齐
            va='bottom',  # 垂直对齐：顶部对齐
            transform=plt.gca().transAxes,  # 使用相对坐标（0-1范围）
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, pad=5)  # 文本背景框
        )
    
    # 设置标题和标签
    plt.title(title, fontsize=15)
    plt.xlabel("t-SNE_dim1", fontsize=12)
    plt.ylabel("t-SNE_dim2", fontsize=12)
    plt.legend(
        title='Type',
        loc='upper right',  # 右上角位置
        bbox_to_anchor=(0.98, 0.98),  # 微调位置，确保在内部
        fontsize=9,        # 适当缩小字体
        title_fontsize=10,
        borderaxespad=0.3  # 调整与坐标轴的距离
    )
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    
    plt.show()
    
    return embeddings_2d

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

            # test_ag_vertex = torch.FloatTensor(test_g['ag_feat']).cuda()
            # test_ag_edge = torch.FloatTensor(test_g['ag_edge_feat']).cuda()
            # test_ag_nh_indices = torch.LongTensor(test_g['ag_nh_indices']).cuda()
            # test_ag_label = torch.LongTensor(test_g['ag_label'])
            # test_ag_indices = [[index, label] for index, label in enumerate(test_g['ag_label'])]
            # np.random.shuffle(down_sample(np.array(test_ag_indices)))
            # test_ag_indices = [index for index, _ in enumerate(test_g['ag_label'])]
            # test_ag_indices = torch.LongTensor(test_ag_indices).cuda()

        # g_preds = model(test_ag_vertex, test_ag_indices)
        # g_preds = model(test_ag_vertex, test_ag_nh_indices, test_ag_indices)
        # 转换 -1 为 0
        test_ag_label = (test_ag_label == 1).int()  # 转换 -1 为 0
        g_preds,u,v = model(test_ag_vertex, test_ag_edge, test_ag_nh_indices, test_ag_indices)
        g_preds = g_preds.data.cpu().numpy()
     
        test_ag_label = test_ag_label.numpy()
        all_preds.append(g_preds)
        all_trues.append(test_ag_label[:,1])

    #     n_samples = combined.shape[0]  # 获取样本数量，假设combined是你的特征数据
    #     labels = test_ag_label[:,1].flatten()
    #     unique_labels = np.unique(labels)
    #     if isinstance(combined, torch.Tensor):
    #         # 从GPU转移到CPU，分离计算图，转换为NumPy数组
    #         combined_np = combined.cpu().detach().numpy()
    #     else:
    #         combined_np = combined  # 如果已经是NumPy数组则直接使用
        
    #     # print(len(labels))
    #     # print(len(unique_labels),len(labels))
    #     if len(unique_labels) >= 2:
    #         sc = silhouette_score(combined_np, labels)
    #         if sc > mx:
    #             mx = sc
    #             comb = combined_np
    #             lab = labels

    #     # print(test_g['PDBID'], g_auc_pr, g_auc_roc)

    # if len(unique_labels) >= 2 and all(np.sum(labels == lab) >= 1 for lab in unique_labels):
    #     sc = silhouette_score(comb, lab)
    #     print(sc)
    #     print(f"轮廓系数（Silhouette Coefficient）: {sc:.4f}")
    # new_labels = ["Non Paratope" if label == 0 else "Paratope" for label in lab]
    # tsne_visualization(
    #     comb, 
    #     new_labels, 
    #     title="Antibody",
    #     palette= ["#344D81","#E81C22"],
    #     sc = mx,
    #     save_path=f"embedding_tsne_{i}.png",
    # )

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

    # test_model = NodeAverageModel()
    # test_model = NodeEdgeAverageModel()
    # test_model = BiLSTMNodeAverageModel()
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
