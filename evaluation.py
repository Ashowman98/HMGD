import numpy as np

from utils import *


def EvaHier_HierarchicalPrecisionAndRecall(y_pre, y, tree):
    FH_sum = 0
    PH_sum = 0
    RH_sum = 0
    num = len(y)
    for i in range(num):
        y_pre_anc = tree_Ancestors(tree, y_pre[i], itself=True)
        y_anc = tree_Ancestors(tree, y[i], itself=True)
        correct = np.intersect1d(y_pre_anc, y_anc)
        PHi = len(correct) / len(y_pre_anc)
        RHi = len(correct) / len(y_anc)
        FHi = 2 * PHi * RHi / (PHi + RHi)
        FH_sum += FHi
        PH_sum += PHi
        RH_sum += RHi
    FH = FH_sum / num
    PH = PH_sum / num
    RH = RH_sum / num

    return FH


def EvaHier_TreeInducedError(y_pre, y, tree):
    num = len(y)
    TIE = 0
    for i in range(num):
        y_pre_anc = tree_Ancestors(tree, y_pre[i], itself=True)
        y_anc = tree_Ancestors(tree, y[i], itself=True)
        y_pre_diff = np.setdiff1d(y_pre_anc, y_anc)
        y_diff = np.setdiff1d(y_anc, y_pre_anc)
        TIE += len(np.append(y_pre_diff, y_diff))
    TIE /= num
    return TIE


def EvaHier_HierarchicalLCAPrecisionAndRecall(y_pre, y, tree):
    FLCA_sum = 0
    PLCA_sum = 0
    RLCA_sum = 0
    num = len(y)
    for i in range(num):
        y_pre_anc = tree_Ancestors(tree, y_pre[i], itself=True)
        y_anc = tree_Ancestors(tree, y[i], itself=True)
        correct = np.intersect1d(y_pre_anc, y_anc)
        PLCAi = 1 / (len(y_pre_anc) - len(correct) + 1)
        RLCAi =  1 / (len(y_anc) - len(correct) + 1)
        FLCAi = 2 * PLCAi * RLCAi / (PLCAi + RLCAi)
        FLCA_sum += FLCAi
        PLCA_sum += PLCAi
        RLCA_sum += RLCAi
    FLCA = FLCA_sum / num
    PLCA = PLCA_sum / num
    RLCA = RLCA_sum / num

    return FLCA