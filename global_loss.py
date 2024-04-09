import numpy as np

from utils import *
from evaluation import EvaHier_TreeInducedError as tie

class Global_Loss():
    def __init__(self, tree, n_out, device):
        root = tree_Root(tree)
        # self.weight = max(tree[1]) - tree[1] + 1
        # self.weight = np.delete(self.weight, root)
        self.weight = torch.tensor(self.get_weight(tree, n_out)).to(torch.float32).to(device)
        self.sigmod = torch.nn.Sigmoid()

    def forward(self, y_predicted, y, targets):
        y_predicted = self.sigmod(y_predicted)
        y_predicted = torch.clamp(y_predicted, min=1e-7, max=1 - 1e-7)
        ce_loss = self.weight[targets, :] * (y * (torch.log(y_predicted)) + (1 - y) * (torch.log(1 - y_predicted)))
        bce = -torch.mean(ce_loss)
        return bce

    def get_weight(self, tree, n_out):
        node_num = tree.shape[1] - 1
        leaf_node = tree_LeafNode(tree)
        weight = np.zeros((len(leaf_node), node_num))
        for i in range(n_out[-1]):
            tree_gap = 0
            Y = tree_Ancestors(tree, i, True)
            Y = np.delete(Y, -1)
            for j in range(len(Y)):
                for k in range(n_out[len(n_out) - j - 1]):
                    weight[i, k+tree_gap] = tie(list([k+tree_gap]), list([Y[j]]), tree)
                weight[i, tree_gap:tree_gap+n_out[len(n_out) - j - 1]] /= max(weight[i, tree_gap:tree_gap+n_out[len(n_out) - j - 1]])
                tree_gap += n_out[len(n_out) - j - 1]
            weight[i, :] += 1
        return weight




