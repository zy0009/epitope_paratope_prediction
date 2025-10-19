import os
import math
import numpy as np
import random
import torch
import pickle
from utils import *
from models import *
from losses import WeightedCrossEntropy
from config import DefaultConfig
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
configs = DefaultConfig()

import os

def contrastive_loss(global_feat, local_feat, temperature=0.1):
    """
    计算全局特征和局部特征间的对比损失
    """
    # 计算相似度矩阵 [B, B]
    global_feat = F.normalize(global_feat, p=2, dim=1)
    local_feat = F.normalize(local_feat, p=2, dim=1)
    logits = torch.mm(global_feat, local_feat.T) / temperature
    
    # 生成标签 (对角线为正样本)
    batch_size = global_feat.size(0)
    labels = torch.arange(batch_size).to(global_feat.device)
    
    # 对称损失计算
    loss_global = F.cross_entropy(logits, labels)
    loss_local = F.cross_entropy(logits.T, labels)
    return (loss_global + loss_local) / 2

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

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',        # 监控验证损失的最小化
        factor=0.95,        # 学习率衰减系数（新lr = lr * factor）
        patience=3,        # 容忍3个epoch无改善后调整
    )
    loss_fn = WeightedCrossEntropy(neg_wt=neg_wt, device=device)
    # 在训练循环中添加学习率跟踪
    current_lr = optimizer.param_groups[0]['lr']  # 初始学习率
    model.train()
    train_losses = []
    val_losses = []
    best_loss = b_loss
    count = 0
    for e in range(epochs):
        print("Running {} epoch".format(e + 1 + re_train_start))
        e_loss = 0.
        for train_g in train_graphs:
            if torch.cuda.is_available():
                train_ag_vertex = torch.FloatTensor(train_g['l_vertex']).cuda()
                train_ag_vertex[:, 43:] = 0
                train_ag_edge = torch.FloatTensor(train_g['l_edge']).cuda()
                train_ag_nh_indices = torch.LongTensor(train_g['l_hood_indices']).cuda()
                train_ag_label = torch.LongTensor(train_g['label']).cuda()
                # train_ag_indices = [[index, label] for index, label in enumerate(train_g['ag_label'])]
                # np.random.shuffle(down_sample(np.array(train_ag_indices)))
                train_indices = [index for index, label in enumerate(train_g['label'])]
                train_ag_indices = torch.LongTensor(train_indices).cuda()

            # 转换 -1 为 0
            train_ag_label[:,1] = (train_ag_label[:,1] == 1).int()  # 转换 -1 为 0

            label_size = len(train_ag_label)
            iters = math.ceil(label_size / batch_size)
            g_loss = 0.
            # 在训练循环中
            for it in range(iters):
                
                start = it * batch_size
                end = start + batch_size
                if end > label_size:
                    end = label_size
                optimizer.zero_grad()
                
                # 前向传播获取三个返回值
                batch_pred, global_proj, local_proj = model(
                    train_ag_vertex, 
                    train_ag_edge, 
                    train_ag_nh_indices, 
                    train_ag_indices[start:end]
                )
  
                # 计算主任务损失
                main_loss = loss_fn.computer_loss(batch_pred, train_ag_label[start:end])
                
                # 计算对比损失
                contrast_loss = contrastive_loss(global_proj, local_proj)

                contrast_weight = 0.1 + 0.9 * (1 - e / epochs)
                # 总损失（权重可调）
                total_loss = main_loss + contrast_weight * contrast_loss
                
                # 反向传播
                total_loss.backward()
                optimizer.step()
                
                # 记录损失
                b_loss = total_loss.item()
                g_loss += b_loss
            g_loss /= iters
            e_loss += g_loss

        e_loss /= len(train_graphs)
        train_losses.append(e_loss)
        with open(os.path.join(model_save_path, 'epi_train_losses{}.txt'.format(num)), 'a+') as f:
            f.write(str(e_loss) + '\n')

        e_loss = 0.
        for val_g in val_graphs:
            if torch.cuda.is_available():
                val_ag_vertex = torch.FloatTensor(val_g['l_vertex']).cuda()
                val_ag_vertex[:, 43:] = 0
                val_ag_edge = torch.FloatTensor(val_g['l_edge']).cuda()
                val_ag_nh_indices = torch.LongTensor(val_g['l_hood_indices']).cuda()
                val_ag_label = torch.LongTensor(val_g['label']).cuda()
                # val_ag_indices = [[index, label] for index, label in enumerate(val_g['ag_label'])]
                # np.random.shuffle(down_sample(np.array(val_ag_indices)))
                val_ag_indices = [index for index, _ in enumerate(val_g['label'])]
                val_ag_indices = torch.LongTensor(val_ag_indices).cuda()

            val_ag_label[:,1] = (val_ag_label[:,1] == 1).int()  # 转换 -1 为 0
            label_size = len(val_ag_label)
            iters = math.ceil(label_size / batch_size)
            g_loss = 0.
            for it in range(iters):
                optimizer.zero_grad()
                start = it * batch_size
                end = start + batch_size
                if end > label_size:
                    end = label_size

                # batch_pred = model(val_ag_vertex, val_ag_indices[start:end])
                # batch_pred = model(val_ag_vertex, val_ag_nh_indices, val_ag_indices[start:end])
                with torch.no_grad():
                    # 只取主预测结果
                    batch_pred, global_proj, local_proj = model(
                        val_ag_vertex,
                        val_ag_edge,
                        val_ag_nh_indices,
                        val_ag_indices[start:end]
                    )
                    main_loss = loss_fn.computer_loss(batch_pred, val_ag_label[start:end])
                    total_loss = main_loss
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
        scheduler.step(val_losses[-1])  # 根据最新验证损失调整
        prev_lr = current_lr  # 保存当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != prev_lr:
            print(f'\n学习率已调整: {prev_lr} -> {current_lr}')
        # else:
        #     count += 1
        #     if count >= 5:
        #         with open(os.path.join(model_save_path, 'epi_train_losses{}.txt'.format(num)), 'w') as f:
        #             for i in train_losses:
        #                 f.write(str(i) + '\n')
        #         with open(os.path.join(model_save_path, 'epi_val_losses{}.txt'.format(num)), 'w') as f:
        #             for j in val_losses:
        #                 f.write(str(j) + '\n')
        #         return None

        # if e % 10 == 0:
        #     print("Epoch {}: train loss {}\tval loss {}".format(e + 1, train_losses[-1], val_losses[-1]))

if __name__ == '__main__':
    seeds = [42,3407,114514,1919810,2025,42,3407,114514,1919810,2025]

    train_path = configs.train_dataset_path
    val_path = configs.val_dataset_path
    print(torch.__version__)
    print("CUDA 可用：", torch.cuda.is_available())
    print("设备数量：", torch.cuda.device_count())

    # data_context
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f,encoding='latin1')

    with open(val_path, 'rb') as f:
        val_data = pickle.load(f,encoding='latin1')

    
    # combined = train_data + val_data
    # np.random.shuffle(combined)
    # train_data = combined[:206]
    # val_data = combined[206:]

    # train_model = NodeAverageModel()
    # train_model = NodeEdgeAverageModel()
    # train_model = BiLSTMNodeAverageModel()
    train_model = BiLSTMNodeEdgeAverageModel()

    # from utils import propress_data
    # train_data = propress_data(train_data)
    # val_data = propress_data(val_data)

    current_experiment = 1
    trained_epochs = 50
    b_v_loss = 999

    for seed in seeds:
        print('experiment:', current_experiment)
        torch.backends.cudnn.deterministic = True

        train_model_dir = os.path.join(configs.save_path, 'model{}.tar'.format(current_experiment))
        if os.path.exists(train_model_dir):
            train_model_sd = torch.load(train_model_dir)
            train_model.load_state_dict(train_model_sd)
            train(train_model, train_data, val_data, current_experiment, 0, b_v_loss)
            current_experiment += 1
        else:
            train(train_model, train_data, val_data, current_experiment)
            current_experiment += 1