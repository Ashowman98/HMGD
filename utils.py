import os
import os.path as p

import numpy
import numpy as np
import math
import torch
import pickle
import random
import sklearn

from torch.utils.data import Dataset
import scipy.io as scio
import scipy.sparse as sp
from sklearn.metrics.pairwise import rbf_kernel
from torchvision import transforms


def setup_seed(seed):
    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def loadData(args):
    data_folder = p.join(args['src_path'], args['dataset'])
    file_path = p.join(data_folder, args['dataset']) + 'Train.mat'
    data_file = scio.loadmat(file_path)
    data = torch.tensor(data_file['data_array'])
    features = data[:, 0:-1]
    labels = data[:, -1] - 1
    tree = data_file['tree'].T
    tree = tree.astype(int)
    tree[0, :] -= 1

    return features, labels, tree


def loadTestData(args):
    data_folder = p.join(args['src_path'], args['dataset'])
    file_path = p.join(data_folder, args['dataset']) + 'Test.mat'
    data_file = scio.loadmat(file_path)
    data = torch.tensor(data_file['data_array'])
    features = data[:, 0:-1]
    labels = data[:, -1] - 1
    tree = data_file['tree'].T
    tree = tree.astype(int)
    tree[0, :] -= 1

    return features, labels, tree


def tree_Root(tree):
    root = np.where(tree[1, :] == 0)[0]
    root.astype(int)
    return root


def tree_InternalNodes(tree):
    middleNode = tree[0, :]
    middleNode = np.unique(middleNode)
    indx = np.where(middleNode == -1)
    middleNode = np.delete(middleNode, indx)
    root = tree_Root(tree)
    root_indx = np.where(middleNode == root)
    middleNode = np.delete(middleNode, root_indx)
    return middleNode


def tree_LeafNode(tree):
    allNode = np.arange(0, tree.shape[1])
    middleNode = tree[0, :]
    middleNode = np.unique(middleNode)
    leafNode = np.setdiff1d(allNode, middleNode)
    return leafNode


def tree_ChildNode(tree, cur_node):
    child_node = np.where(tree[0, :] == cur_node)
    return child_node[0]


def tree_CurLeaf(tree, cur_node):
    descendant = tree_ChildNode(tree, cur_node)
    leafnode = tree_LeafNode(tree)
    cur_leaf = np.empty(0).astype(int)
    while len(descendant) != 0:
        temp_node = descendant[0]
        descendant = np.delete(descendant, range(0, 1), axis=0)
        if temp_node in leafnode:
            cur_leaf = numpy.append(cur_leaf, temp_node)
        else:
            descendant = numpy.append(descendant, tree_ChildNode(tree, temp_node))
    return cur_leaf


def tree_Ancestors(tree, cur_node, itself=False):
    ancestor = np.empty(0).astype(int)
    if itself:
        ancestor = np.append(ancestor, cur_node)
    temp_node = tree[0, cur_node]
    while temp_node != -1:
        ancestor = np.append(ancestor, temp_node)
        temp_node = tree[0, temp_node]

    return ancestor




def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def re_label(labels, tree, ratio):
    re_labels = torch.tensor(labels)
    labels_no = np.unique(re_labels)
    for i in range(len(labels_no)):
        idx = np.where(re_labels == labels_no[i])[0]
        re_num = math.floor(len(idx) * ratio)
        re_idx = random.sample(range(0, len(idx)), re_num)
        # re_idx = range(0, re_num-1)
        re_labels[idx[re_idx]] = torch.tensor(tree[0][labels[idx[re_idx]].to(torch.int)]).to(torch.double)

    return re_labels



class MLLDataset(Dataset):
    def __init__(self, args, istrain=True, pre=False):
        super(MLLDataset, self).__init__()
        if istrain:
            self.features, self.labels, self.tree = loadData(args)
        else:
            self.features, self.labels, self.tree = loadTestData(args)
        if pre:
            self.features = torch.tensor(sklearn.preprocessing.StandardScaler().fit_transform(self.features))
        if args['relabel'] > 0 and istrain==True:
            self.labels = re_label(self.labels, self.tree, args['relabel'])



    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]




