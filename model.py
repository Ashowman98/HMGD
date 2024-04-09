import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import tree_InternalNodes, tree_Root, tree_ChildNode, tree_LeafNode, tree_CurLeaf


class HLE(nn.Module):
    def __init__(self, n_in, n_hidden, tree, keep_prob=1.0):
        super(HLE, self).__init__()
        self.InternalNodes = tree_InternalNodes(tree)
        self.root = tree_Root(tree)
        self.leafnode = tree_LeafNode(tree)
        self.noLeafNode = np.append(self.InternalNodes, self.root)
        self.tree = tree
        self.level_num = max(tree[1, :])
        self.level_target = [[] for _ in range(self.level_num)]
        self.n_out = [[] for _ in range(self.level_num)]
        for level in range(self.level_num):
            self.level_target[level] = np.where(tree[1, :] == level+1)[0]
            self.n_out[level] = len(self.level_target[level])
        self.mlist = nn.ModuleList([localmodel(n_in, n_hidden, self.n_out[level], keep_prob) for level in range(self.level_num)])
        self.final_linear = nn.Linear(self.level_num, 1)
        self.final_bn = nn.BatchNorm1d(self.n_out[-1])
        nn.init.constant(self.final_linear.weight, 1)
        nn.init.constant(self.final_linear.bias, 0)

    def forward(self, inputs, device):
        out = [[] for _ in range(self.level_num)]
        add_pro = []
        for level in range(self.level_num):
            out[level] = (self.mlist[level](inputs))
            if level != self.level_num-1:
                add_pro.append(torch.zeros((inputs.shape[0], self.n_out[-1])).to(device))
                for i in range(self.n_out[level]):
                    cur_node = self.level_target[level][i]
                    cur_leaf = tree_CurLeaf(self.tree, cur_node)
                    add_pro[level][:, cur_leaf] += torch.unsqueeze(out[level][:, i], dim=1)
        final_out = torch.unsqueeze(out[self.level_num-1], dim=2)
        for level in range(self.level_num-1):
            add_pro[level] = torch.unsqueeze(add_pro[level], dim=2)
            final_out = torch.cat((final_out, add_pro[level]), dim=2)
        final_out = self.final_linear(final_out)
        final_out = torch.squeeze(final_out, dim=2)
        final_out = self.final_bn(final_out)

        return out, final_out

    def forward_test(self, inputs, inputs_flat):
        out = torch.empty(0)
        out_global = self.globalmodel(inputs_flat)
        for n in range(inputs.shape[0]):
            cur_node = self.root[0]
            leaf_num = len(self.leafnode)
            while True:
                cur_fea = inputs[n, :]
                pre = self.mlist[cur_node-leaf_num](cur_fea)
                max_prob_node = torch.argmax(pre)
                cur_chi_node = tree_ChildNode(self.tree, cur_node)
                cur_node = cur_chi_node[0][max_prob_node]
                if cur_node not in self.noLeafNode:
                    out = torch.cat((out, torch.tensor([cur_node])))
                    break
        return out


class localmodel(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, keep_prob=1.0):
        super(localmodel, self).__init__()
        self.n_out = n_out
        self.layer1 = nn.Sequential(nn.Linear(n_in, n_hidden),
                                    nn.BatchNorm1d(n_hidden),
                                    nn.ELU(),
                                    nn.Dropout(1 - keep_prob))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                                    nn.BatchNorm1d(n_hidden),
                                    nn.ELU(),
                                    nn.Dropout(1 - keep_prob))
        self.fc_out = nn.Sequential(nn.Linear(n_hidden, n_out),
                                    nn.BatchNorm1d(n_out))
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)

    def forward(self, inputs):
        out1 = self.layer1(inputs)
        out2 = self.layer2(out1)
        out = self.fc_out(out2)
        return out




class VAE_Encoder(nn.Module):
    def __init__(self,n_in,n_hidden,n_out,keep_prob=1.0):
        super(VAE_Encoder,self).__init__()
        self.n_out = n_out
        self.layer1 = nn.Sequential(nn.Linear(n_in,n_hidden),
                                    nn.ELU(),
                                    nn.Dropout(1-keep_prob))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden,n_hidden),
                                    nn.Tanh(),
                                    nn.Dropout(1-keep_prob))
        self.fc_out = nn.Linear(n_hidden,n_out*2)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)

    def forward(self,inputs,eps=1e-8):
        h0 = self.layer1(inputs)
        h1 = self.layer2(h0)
        out = self.fc_out(h1)
        mean = out[:,:self.n_out]
        std = F.softplus(out[:,self.n_out:]) + eps
        return (mean,std)

class VAE_Decoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_label, keep_prob=1.0):
        super(VAE_Decoder,self).__init__()
        self.n_label = n_label
        self.layer1 = nn.Sequential(nn.Linear(n_in,n_hid),
                nn.Tanh(),
                nn.Dropout(1-keep_prob))
        self.layer2 = nn.Sequential(nn.Linear(n_hid,n_hid),
                nn.ELU(),
                nn.Dropout(1-keep_prob))
        self.fc_out = nn.Linear(n_hid,n_out)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)

    def forward(self, z):
        h0 = self.layer1(z)
        h1 = self.layer2(h0)
        features_hat = self.fc_out(h1)
        labels_hat = F.sigmoid(z[:, -self.n_label:])
        return features_hat, labels_hat

class VAE_Bernulli_Decoder(nn.Module):
    def __init__(self,n_in,n_hidden,n_out,keep_prob=1.0):
        super(VAE_Bernulli_Decoder,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(n_in,n_hidden),
                                    nn.Tanh(),
                                    nn.Dropout(1-keep_prob))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden,n_hidden),
                                    nn.ELU(),
                                    nn.Dropout(1-keep_prob))
        self.fc_out = nn.Linear(n_hidden,n_out)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)

    def forward(self,inputs):
        h0 = self.layer1(inputs)
        h1 = self.layer2(h0)
        out = F.sigmoid(self.fc_out(h1))
        return out

class VAE_Gauss_Decoder(nn.Module):
    def __init__(self,n_in,n_hidden,n_out,keep_prob=1.0):
        super(VAE_Gauss_Decoder,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(n_in,n_hidden),
                                    nn.Tanh(),
                                    nn.Dropout(1-keep_prob))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden,n_hidden),
                                    nn.ELU(),
                                    nn.Dropout(1-keep_prob))
        self.fc_mean = nn.Linear(n_hidden,n_out)
        self.fc_var = nn.Linear(n_hidden,n_out)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)

    def forward(self,inputs):
        h0 = self.layer1(inputs)
        h1 = self.layer2(h0)
        mean = self.fc_mean(h1)
        var = F.softplus(self.fc_var(h1))
        return mean,var


class VGAE_Encoder(nn.Module):
    def __init__(self, adj, n_in, n_hid, n_out):
        super(VGAE_Encoder,self).__init__()
        self.n_out = n_out
        self.base_gcn = GraphConv(n_in, n_hid, adj)
        self.gcn = GraphConv(n_hid, n_out*2, adj, activation=lambda x:x)

    def forward(self, x):
        hidden = self.base_gcn(x)
        out = self.gcn(hidden)
        mean = out[:,:self.n_out]
        std = out[:,self.n_out:]
        return (mean,std)


class VGAE_Decoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_label, keep_prob=1.0):
        super(VGAE_Decoder,self).__init__()
        self.n_label = n_label
        self.layer1 = nn.Sequential(nn.Linear(n_in,n_hid),
                nn.Tanh(),
                nn.Dropout(1-keep_prob))
        self.layer2 = nn.Sequential(nn.Linear(n_hid,n_hid),
                nn.ELU(),
                nn.Dropout(1-keep_prob))
        self.fc_out = nn.Linear(n_hid,n_out)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)

    def forward(self, z):
        h0 = self.layer1(z)
        h1 = self.layer2(h0)
        features_hat = F.sigmoid(self.fc_out(h1))
        labels_hat = F.sigmoid(z[:, -self.n_label:])
        adj_hat = dot_product_decode(z)
        return features_hat, labels_hat, adj_hat


class GraphConv(nn.Module):
    def __init__(self, n_in, n_out, adj, activation = F.relu, **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.weight = glorot_init(n_in, n_out)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x,self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)
