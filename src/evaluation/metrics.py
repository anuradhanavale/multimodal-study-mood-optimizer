import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

def evaluate(y, proba, n_classes=3):
    y = np.asarray(y); pred = proba.argmax(1)
    onehot = np.eye(n_classes)[y]
    return {"macro_f1": float(f1_score(y, pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y, pred, average="macro", zero_division=0)),
            "brier": float(((proba - onehot) ** 2).sum(1).mean()),
            "confusion": confusion_matrix(y, pred, labels=list(range(n_classes))).tolist()}
