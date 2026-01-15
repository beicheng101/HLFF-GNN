import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter, Linear, Sequential, LogSoftmax


class HLFFGNN(Module):
    def __init__(self, n, nclass, nfeat, nlayer, lambda_1, lambda_2, lambda_3, alpha, dropout):
        super(HLFFGNN, self).__init__()
        self.n = n
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.alpha = alpha
        self.nclass = nclass
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.dropout = dropout
        self.w1 = Parameter(torch.FloatTensor(nfeat, nfeat), requires_grad=True)
        # 根据拼接个数改数字
        self.w2 = Sequential(Linear(2 * nfeat, nclass), LogSoftmax(dim=1))
        self.params1 = [self.w1]
        self.params2 = list(self.w2.parameters())
        self.laplacian = None
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.uniform_(self.w1, 0, 1)

    def forward(self, feat, adj):
        if self.laplacian is None:
            n = adj.shape[0]
            indices = torch.Tensor([list(range(n)), list(range(n))])
            values = torch.FloatTensor([1.0] * n)
            eye = torch.sparse_coo_tensor(indices=indices, values=values, size=[n, n]).to(adj.device)
            self.laplacian = eye - adj

        lap_low = self.laplacian     # L low
        lap_high = adj                # L high

        y: Tensor = torch.rand(self.n, self.nfeat).to(adj.device)
        y_low = torch.mm(adj, y)        # Y low
        y_high = torch.mm(lap_low, y)   # Y high
        z1: Tensor = feat
        z2: Tensor = feat
        for i in range(self.nlayer):
            feat = F.dropout(feat, self.dropout, training=self.training)

            # 原始HLFF-GNN,只引入L low和L high

            """sig(z1wz1T)*Y+sig(z2wz2T)*Y
            temp = torch.mm(self.w1, z1.t())
            temp = torch.mm(z1, temp)
            temp = torch.sigmoid(temp)

            temp = torch.mm(temp, y)
            y_n = feat - self.lambda_3 * temp  # 改成1/2？
            # Z1
            temp = torch.mm(y.t(), z1)
            temp = torch.mm(y, temp)
            temp = torch.sigmoid(temp)
            temp1 = torch.mm(lap_high, z1)
            z_n1 = temp1 - (self.lambda_3 / self.lambda_1) * temp

            feat = F.dropout(feat, self.dropout, training=self.training)
            temp = torch.mm(self.w1, z2.t())
            temp = torch.mm(z2, temp)
            temp = torch.sigmoid(temp)
            temp = torch.mm(temp, y)
            y_n = feat - self.lambda_3 * temp  # 改成1/2？
            # Z2
            temp = torch.mm(y.t(), z2)
            temp = torch.mm(y, temp)
            temp = torch.sigmoid(temp)
            temp1 = torch.mm(lap_low, z2)
            z_n2 = temp1 - (self.lambda_3 / self.lambda_2) * temp
            # z1+z2 拼接
            y = y_n
            z = z_n2"""

            # 引入Y low和Y high，对HLFF-GNN做进一步改进

            temp = torch.mm(self.w1, z1.t())
            temp = torch.mm(z1, temp)
            temp = torch.sigmoid(temp)
            temp1 = torch.mm(temp, y)
            temp = torch.mm(self.w1, z2.t())
            temp = torch.mm(z2, temp)
            temp = torch.sigmoid(temp)
            temp2 = torch.mm(temp, y)
            y_n = feat - self.lambda_3 * (temp1 + temp2)
            # Z1
            temp = torch.mm(y_high.t(), z1)
            temp = torch.mm(y_high, temp)
            temp = torch.sigmoid(temp)
            temp1 = torch.mm(lap_high, z1)
            z_n1 = temp1 - (self.lambda_3 / self.lambda_1) * temp
            # Z2
            temp = torch.mm(y_low.t(), z2)
            temp = torch.mm(y_low, temp)
            temp = torch.sigmoid(temp)
            temp1 = torch.mm(lap_low, z2)
            z_n2 = temp1 - (self.lambda_3 / self.lambda_2) * temp

            # z1+z2 拼接
            y = y_n
            z1 = z_n1
            z2 = z_n2
            z = self.alpha * z1 + (1 - self.alpha) * z2

        y = F.normalize(y, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        p = torch.cat((y, z), dim=1)
        p = F.dropout(p, self.dropout, training=self.training)
        return self.w2(p)