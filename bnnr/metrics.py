"""
Evaluation metrics for drug-disease association prediction.

Contains:
  - getPerfMetricROCcompute()  -- vectorized ROC/PR computation (AUROC, AUPR, etc.)
  - compute_topk_metrics()     -- Top-K precision, recall, and hits
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid


def getPerfMetricROCcompute(predictList, trueList, poslabel, plotOption):
    """
    Vectorized ROC/PR computation (~50x faster than loop-based version).

    Parameters
    ----------
    predictList : array-like, model predictions
    trueList : array-like, ground-truth labels
    poslabel : int/float, positive class label value
    plotOption : int, 1=plot ROC/PR curves, 0=no plot

    Returns
    -------
    tbScalar : DataFrame, AUROC/AUPRC and threshold metrics
    tbVec : DataFrame, per-threshold metrics
    AUC : float, area under ROC curve
    AUPR : float, area under PR curve
    Acc, Sen, Spe, Pre : ndarray, per-threshold metrics
    """
    predictList = np.asarray(predictList).ravel()
    trueList = np.asarray(trueList).ravel()

    Npos = int(np.sum(trueList == poslabel))
    Nneg = len(trueList) - Npos
    len_pred = len(predictList)

    # Sort predictions in descending order
    ord_idx = np.argsort(predictList)[::-1]
    rank_t = np.empty(len_pred, dtype=np.float64)
    rank_t[ord_idx] = np.arange(1.0, len_pred + 1.0)
    predictList_sorted = 1.0 - rank_t / len_pred

    pos_mask = (trueList == poslabel)
    neg_mask = ~pos_mask

    # Sample thresholds
    n_thresholds = min(1000, len_pred)
    threshold_indices = np.linspace(0, len_pred - 1, n_thresholds, dtype=int)

    # Cumulative counts in sorted order
    sorted_true = trueList[ord_idx]
    cum_pos = np.cumsum(sorted_true == poslabel)
    cum_neg = np.cumsum(sorted_true != poslabel)

    tp = cum_pos[threshold_indices].astype(np.float64)
    fp = cum_neg[threshold_indices].astype(np.float64)
    fn = Npos - tp
    tn = Nneg - fp

    denom_pos = tp + fn
    denom_neg = tn + fp
    denom_prec = tp + fp

    Sen = np.where(denom_pos == 0, 1.0, tp / denom_pos)
    Spe = np.where(denom_neg == 0, 1.0, tn / denom_neg)
    Pre = np.where(denom_prec == 0, 1.0, tp / denom_prec)
    Acc = (tn + tp) / len_pred

    # Optimal cut point
    diff = np.abs(Npos - (tp + fp))
    ind_cut = int(np.argmin(diff))

    # Pad arrays
    Sen = np.concatenate([[0], Sen])
    Spe = np.concatenate([[1], Spe])
    Pre = np.concatenate([[Pre[0]], Pre])
    acc_init = Nneg / len_pred
    Acc = np.concatenate([[acc_init], Acc])
    ind_cut += 1

    FPR = 1 - Spe
    AUC = abs(trapezoid(Sen, FPR))
    AUPR = abs(trapezoid(Pre, Sen))

    tbScalar = pd.DataFrame({
        'AUROC': [AUC],
        'AUPRC': [AUPR],
        'Acc': [Acc[ind_cut]],
        'Sen': [Sen[ind_cut]],
        'Spe': [Spe[ind_cut]],
        'Pre': [Pre[ind_cut]],
        'mAcc': [np.mean(Acc)],
        'mSen': [np.mean(Sen)],
        'mSpe': [np.mean(Spe)],
        'mPre': [np.mean(Pre)]
    })

    tbVec = pd.DataFrame({
        'Acc': Acc,
        'Pre': Pre,
        'TPR_Rec_Sen': Sen,
        'FPR': FPR,
        'Spe': Spe
    })

    if plotOption == 1:
        plt.figure(figsize=(6, 6))
        plt.plot(FPR, Sen, color='#1f77b4')
        plt.axis([-0.01, 1.00, 0, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (AUROC = {AUC:.4f})')
        plt.grid(alpha=0.3, linestyle='--')

        plt.figure(figsize=(6, 6))
        plt.plot(Sen, Pre, color='#ff7f0e')
        plt.axis([0, 1.01, 0, 1.01])
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision')
        plt.title(f'PR Curve (AUPR = {AUPR:.4f})')
        plt.grid(alpha=0.3, linestyle='--')
        plt.show()

    return tbScalar, tbVec, AUC, AUPR, Acc, Sen, Spe, Pre


def compute_topk_metrics(scores, labels, ks=(10, 20)):
    """
    Compute Top-K precision, recall, and hits metrics.

    Parameters
    ----------
    scores : array-like, prediction scores (higher = more likely positive), 1D
    labels : array-like, ground-truth labels (1 = positive, 0 = negative), 1D
    ks : tuple/list, K values to evaluate

    Returns
    -------
    results : dict
        Keys: P@{k}, R@{k}, Hits@{k} for each k in ks.
    """
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel().astype(int)

    if scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must have same length")

    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]
    n_pos = int(np.sum(labels == 1))
    results = {}

    for k in ks:
        k_eff = min(int(k), len(labels_sorted))
        if k_eff <= 0:
            results[f'P@{k}'] = 0.0
            results[f'R@{k}'] = 0.0
            results[f'Hits@{k}'] = 0
            continue

        topk_labels = labels_sorted[:k_eff]
        hits = int(np.sum(topk_labels == 1))
        results[f'P@{k}'] = hits / k_eff
        results[f'R@{k}'] = hits / n_pos if n_pos > 0 else 0.0
        results[f'Hits@{k}'] = hits

    return results
