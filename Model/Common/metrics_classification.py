"""ตัวชี้วัดงาน Classification: accuracy, precision, recall, F1, ROC-AUC, log loss"""

import numpy as np


def logloss(y_true, proba, eps=1e-15):
    p = np.clip(np.asarray(proba, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _as_pred(y_true, proba, threshold):
    pred = (np.asarray(proba, dtype=float) >= threshold).astype(int)
    return np.asarray(y_true, dtype=int).ravel(), pred.ravel()


def _resolve_threshold(y_true, proba, threshold, metric="f1"):
    if threshold is None:
        return best_threshold(y_true, proba, metric=metric)[0]
    return float(threshold)


def accuracy(y_true, proba, threshold=0.5):
    threshold = _resolve_threshold(y_true, proba, threshold, metric="accuracy")
    y, pred = _as_pred(y_true, proba, threshold)
    return float(np.mean(pred == y))


def confusion_matrix(y_true, proba, threshold=0.5):
    threshold = _resolve_threshold(y_true, proba, threshold, metric="f1")
    y, pred = _as_pred(y_true, proba, threshold)
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tp = int(np.sum((pred == 1) & (y == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def precision(y_true, proba, threshold=0.5):
    threshold = _resolve_threshold(y_true, proba, threshold, metric="precision")
    cm = confusion_matrix(y_true, proba, threshold)
    tp, fp = cm[1, 1], cm[0, 1]
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(y_true, proba, threshold=0.5):
    threshold = _resolve_threshold(y_true, proba, threshold, metric="recall")
    cm = confusion_matrix(y_true, proba, threshold)
    tp, fn = cm[1, 1], cm[1, 0]
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1(y_true, proba, threshold=0.5):
    threshold = _resolve_threshold(y_true, proba, threshold, metric="f1")
    cm = confusion_matrix(y_true, proba, threshold)
    tp, fp, fn = cm[1, 1], cm[0, 1], cm[1, 0]
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0


def _tie_aware_ranks(scores):
    n = len(scores)
    order = np.argsort(scores, kind="stable")
    s_sorted = scores[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def roc_auc(y_true, proba):
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _tie_aware_ranks(p)
    r_pos = float(ranks[y == 1].sum())
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true, proba):
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-p, kind="stable")
    y_sorted, p_sorted = y[order], p[order]
    group_ends = np.append(np.where(np.diff(p_sorted) != 0)[0], len(y_sorted) - 1)
    tp = np.cumsum(y_sorted)[group_ends].astype(float)
    fp = np.cumsum(1 - y_sorted)[group_ends].astype(float)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / n_pos

    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def roc_curve_scratch(y_true, proba):
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    thresholds = np.unique(p)[::-1]
    tpr, fpr = [0.0], [0.0]
    for thr in thresholds:
        pred = p >= thr
        tpr.append(float(np.sum(pred & (y == 1)) / n_pos) if n_pos else 0.0)
        fpr.append(float(np.sum(pred & (y == 0)) / n_neg) if n_neg else 0.0)
    return np.array(fpr), np.array(tpr), np.concatenate([thresholds, [thresholds[-1] - 1e-12]])


def precision_recall_curve_scratch(y_true, proba):
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    n_pos = int(y.sum())
    thresholds = np.unique(p)[::-1]
    prec, rec = [], []
    for thr in thresholds:
        pred = p >= thr
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        prec.append(float(tp / (tp + fp)) if (tp + fp) else 1.0)
        rec.append(float(tp / n_pos) if n_pos else 0.0)
    return np.array(prec), np.array(rec), thresholds


def classification_metrics(y_true, proba, threshold=0.5, metric="f1"):
    auto = threshold is None
    if auto:
        threshold = _resolve_threshold(y_true, proba, threshold, metric=metric)
    cm = confusion_matrix(y_true, proba, threshold)
    return {
        "logloss": logloss(y_true, proba),
        "accuracy": accuracy(y_true, proba, threshold),
        "precision": precision(y_true, proba, threshold),
        "recall": recall(y_true, proba, threshold),
        "f1": f1(y_true, proba, threshold),
        "roc_auc": roc_auc(y_true, proba),
        "pr_auc": average_precision(y_true, proba),
        "threshold": float(threshold),
        "threshold_auto_selected": bool(auto),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def best_threshold(y_true, proba, metric="f1"):
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    grid = np.unique(np.concatenate([np.linspace(0.001, 0.999, 999), p]))
    best_thr, best_score = 0.5, -1.0
    for thr in grid:
        if metric == "f1":
            score = f1(y, p, thr)
        elif metric == "precision":
            score = precision(y, p, thr)
        elif metric == "recall":
            score = recall(y, p, thr)
        else:
            score = accuracy(y, p, thr)
        if score > best_score:
            best_score, best_thr = score, float(thr)
    return best_thr, best_score


def evaluate_on_test(y_val, proba_val, y_test, proba_test, metric="f1", reference_threshold=0.5):
    thr, val_score = best_threshold(y_val, proba_val, metric=metric)
    out = {"chosen_threshold": float(thr), "val_" + metric: float(val_score)}
    for k, v in classification_metrics(y_test, proba_test, threshold=thr).items():
        out["test_" + k] = v
    if reference_threshold is not None:
        out["test_accuracy@0.5"] = accuracy(y_test, proba_test, reference_threshold)
        out["test_precision@0.5"] = precision(y_test, proba_test, reference_threshold)
        out["test_recall@0.5"] = recall(y_test, proba_test, reference_threshold)
        out["test_f1@0.5"] = f1(y_test, proba_test, reference_threshold)
    return out
