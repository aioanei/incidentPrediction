import numpy as np
import matplotlib
matplotlib.use("Agg")          # no GUI needed
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
)


def print_report(y_true, y_pred, y_prob, model_name: str = "Model"):
    print(f"{model_name} - Evaluation Results")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Incident"]))
    auc = roc_auc_score(y_true, y_prob)
    print(f"  ROC-AUC: {auc:.4f}")
    return auc


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str = None):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Incident"])
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()


def plot_roc_curve(y_true, y_prob, model_name: str, save_path: str = None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} - ROC Curve")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()


def plot_precision_recall(y_true, y_prob, model_name: str, save_path: str = None):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{model_name} - Precision-Recall Curve")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()


def threshold_analysis(y_true, y_prob, model_name: str, save_path: str = None):
    thresholds = np.arange(0.05, 1.0, 0.05)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        # avoid warnings on edge cases
        if preds.sum() == 0:
            precisions.append(0)
            recalls.append(0)
            f1s.append(0)
            continue
        from sklearn.metrics import precision_score, recall_score
        precisions.append(precision_score(y_true, preds, zero_division=0))
        recalls.append(recall_score(y_true, preds, zero_division=0))
        f1s.append(f1_score(y_true, preds, zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, precisions, label="Precision", marker="o", markersize=3)
    ax.plot(thresholds, recalls, label="Recall", marker="s", markersize=3)
    ax.plot(thresholds, f1s, label="F1", marker="^", markersize=3)
    ax.set_xlabel("Alert Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} - Threshold vs Precision/Recall")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()

    # print best threshold by F1
    best_idx = np.argmax(f1s)
    best_t = thresholds[best_idx]
    print(f"[{model_name}] Best threshold by F1: {best_t:.2f}  "
          f"(P={precisions[best_idx]:.3f}, R={recalls[best_idx]:.3f}, "
          f"F1={f1s[best_idx]:.3f})")
    return best_t
