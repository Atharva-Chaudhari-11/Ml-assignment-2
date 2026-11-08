"""
metrics_utils.py - one function that computes all 6 required metrics.
Used by BOTH train_models.py and app.py so numbers always match.

Notation: TP/TN = correct positive/negative predictions,
          FP/FN = wrong positive/negative predictions.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)


def compute_metrics(y_true, y_pred, y_proba):
    n_classes = len(np.unique(y_true))

    # Binary -> 'binary' average (looks only at the positive class).
    # Multi-class -> 'weighted' average (each class's score weighted by its size).
    avg = "binary" if n_classes == 2 else "weighted"

    # AUC needs probabilities, not hard labels.
    if n_classes == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")

    return {
        # Accuracy  = (TP + TN) / (TP + TN + FP + FN)
        "Accuracy": accuracy_score(y_true, y_pred),
        # AUC = area under the ROC curve (TPR vs FPR). 1.0 perfect, 0.5 random guess.
        "AUC": auc,
        # Precision = TP / (TP + FP)   "of everything I called positive, how much was right?"
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        # Recall    = TP / (TP + FN)   "of all real positives, how many did I catch?"
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        # F1 = 2 * Precision * Recall / (Precision + Recall)   (harmonic mean)
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        # ranges -1..+1, stays honest even when classes are imbalanced.
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
