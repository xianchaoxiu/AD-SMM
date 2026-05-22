"""
评估指标
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def compute_metrics(y_true, y_pred, y_score=None):

    yt = np.array(y_true)
    yp = np.array(y_pred)
    yt_01 = (yt + 1) // 2  # +1->1, -1->0
    yp_01 = (yp + 1) // 2

    metrics = {
        'accuracy': accuracy_score(yt_01, yp_01),
        'precision': precision_score(yt_01, yp_01, zero_division=0),
        'recall': recall_score(yt_01, yp_01, zero_division=0),
        'f1': f1_score(yt_01, yp_01, zero_division=0),
    }

    if y_score is not None:
        try:
            metrics['auc'] = roc_auc_score(yt_01, np.array(y_score))
        except ValueError:
            metrics['auc'] = 0.0

    return metrics
