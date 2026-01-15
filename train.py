import time

import numpy as np
import torch
import torch.nn.functional as F
import argparse
from hps import get_hyper_param
from model.HLFFGNN import HLFFGNN
from util import load_dataset, root, get_mask, get_accuracy, set_seed
from torch_geometric.utils import is_undirected, to_undirected, homophily
import random

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
args = parser.parse_args()
set_seed(0xC0FFEE)
epochs = 1000
patience = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = root + "/checkpoint"
feat, label, n, nfeat, nclass, adj = load_dataset(args.dataset, norm=True, device=device)
hp = get_hyper_param(args.dataset)

adj = adj.to_dense()
edge_index = adj.nonzero().t().contiguous()
h = homophily(edge_index, label, method="node")


def create_new_adjacency_matrix(label, n, nfeat, adj, nclass, h):
    # 初始化新的邻居字典
    new_neighbors_dict = {}

    # 获取每个类别的节点集合
    label_sets = {}
    for label_value in set(label.cpu().numpy()):
        label_sets[label_value] = [i for i, l in enumerate(label.cpu().numpy()) if l == label_value]

    # 获取所有节点的索引
    all_nodes = list(range(n))

    # 预先计算每个节点的同类和异类节点集合
    same_class_neighbors_dict = {}
    different_class_neighbors_dict = {}

    for v in range(n):
        # 获取节点 v 的一阶邻居
        neighbors = adj[v].nonzero().squeeze().cpu().numpy()

        # 处理零维数组的情况
        if neighbors.ndim == 0:
            neighbors = []

        # 获取节点 v 的同类节点
        same_class_nodes = label_sets[label[v].item()]
        same_class_neighbors = [i for i in neighbors if label[i].item() == label[v].item()]
        same_class_neighbors_dict[v] = same_class_neighbors

        # 获取节点 v 的异类节点
        different_class_nodes = [i for i in neighbors if label[i].item() != label[v].item()]
        different_class_neighbors_dict[v] = different_class_nodes

    for v in range(n):
        # 获取节点 v 的同类和异类节点
        same_class_neighbors = same_class_neighbors_dict[v]
        different_class_neighbors = different_class_neighbors_dict[v]

        # 获取节点 v 的一阶邻居
        neighbors = adj[v].nonzero().squeeze().cpu().numpy()

        # 处理零维数组的情况
        if neighbors.ndim == 0:
            neighbors = []

        # 计算需要选取的节点数量
        num_same_class_new_neighbors = int(h * len(neighbors))
        num_different_class_new_neighbors = len(neighbors) - num_same_class_new_neighbors

        # 补充同类节点
        if num_same_class_new_neighbors > len(same_class_neighbors):
            additional_same_class_neighbors = random.choices(
                [i for i in label_sets[label[v].item()] if i not in same_class_neighbors],
                k=num_same_class_new_neighbors - len(same_class_neighbors)
            )
            same_class_neighbors.extend(additional_same_class_neighbors)

        # 补充异类节点
        if num_different_class_new_neighbors > len(different_class_neighbors):
            additional_different_class_neighbors = random.choices(
                [i for i in all_nodes if
                 i != v and label[i].item() != label[v].item() and i not in different_class_neighbors],
                k=num_different_class_new_neighbors - len(different_class_neighbors)
            )
            different_class_neighbors.extend(additional_different_class_neighbors)

        # 合并同类和异类节点
        new_neighbors_dict[v] = same_class_neighbors + different_class_neighbors

    # 初始化新的邻接矩阵
    new_A = torch.zeros_like(adj)  # 使用稠密张量初始化

    # 根据新的邻居关系填充新的邻接矩阵
    for v, new_neighbors in new_neighbors_dict.items():
        for neighbor in new_neighbors:
            new_A[v][neighbor] = 1
            new_A[neighbor][v] = 1  # 如果是无向图

    # 将新的邻接矩阵转换为稀疏张量
    new_A_sparse = new_A.to_sparse()

    return new_A_sparse.to(device), nfeat, nclass, n

# 使用新函数获取新的邻接矩阵和相关参数
new_adj, nfeat, nclass, n = create_new_adjacency_matrix(label, n, nfeat, adj, nclass, h)


def train(model, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    result = model(feat=feat, adj=new_adj)
    loss = F.nll_loss(result[train_mask], label[train_mask])
    loss.backward()
    optimizer.step()
    return get_accuracy(result[train_mask], label[train_mask]), loss.item()


def test(model, test_mask):
    model.eval()
    with torch.no_grad():
        result = model(feat=feat, adj=new_adj)
        loss = F.nll_loss(result[test_mask], label[test_mask].to(device))
        return get_accuracy(result[test_mask], label[test_mask]), loss.item()


def validate(model, val_mask) -> float:
    model.eval()
    with torch.no_grad():
        result = model(feat=feat, adj=new_adj)
        return get_accuracy(result[val_mask], label[val_mask])


def run():
    train_mask, test_mask, val_mask = get_mask(label, 0.6, 0.2, device=device)
    model = HLFFGNN(
        n=n,
        nclass=nclass,
        nfeat=nfeat,
        nlayer=hp["layer"],
        lambda_1=hp["lambda_1"],
        lambda_2=hp["lambda_2"],
        lambda_3=hp["lambda_3"],
        dropout=hp["dropout"],
        alpha=hp["alpha"],
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {'params': model.params1, 'weight_decay': hp["wd1"]},
            {'params': model.params2, 'weight_decay': hp["wd2"]}
        ],
        lr=hp["lr"]
    )
    checkpoint_file = "{}/{}-{}.pt".format(checkpoint_path, model.__class__.__name__, args.dataset)
    tolerate = 0
    best_loss = 100
    for epoch in range(epochs):
        if tolerate >= patience:
            break
        train_acc, train_loss = train(model, optimizer, train_mask)
        test_acc, test_loss = test(model, test_mask)
        if train_loss < best_loss:
            tolerate = 0
            best_loss = train_loss
        else:
            tolerate += 1
        message = "Epoch={:<4} | Tolerate={:<3} | Train_acc={:.4f} | Train_loss={:.4f} | Test_acc={:.4f} | Test_loss={:.4f}".format(
            epoch,
            tolerate,
            train_acc,
            train_loss,
            test_acc,
            test_loss
        )
        print(message)

    val_acc = validate(model, val_mask)
    torch.save(model.state_dict(), checkpoint_file)
    print("Validate accuracy {:.4f}.".format(val_acc))
    return val_acc


if __name__ == '__main__':
    run()
