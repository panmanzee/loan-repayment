"""
Shared evaluation metrics for the CLASSIFICATION task (target: not.fully.paid), written from scratch.
This is the classification twin of Common/metrics.py (which holds the regression metrics).

Every function takes plain numpy arrays, so it works with any model (the scratch XGBoost in
ClassificationAlgorithms/XGB.py or anything else).

Loss        : logloss (binary cross-entropy) -- the classification LOSS
Threshold   : accuracy, confusion_matrix, precision, recall, f1
Rank-based  : average_precision (PR-AUC), roc_auc (ROC-AUC) + the two curve helpers used for plots

Why two AUCs: the target is imbalanced (16% positives), so a random classifier already scores
ROC-AUC 0.5 but PR-AUC equal to the positive rate (0.16). PR-AUC is therefore the honest headline
number for this data set; ROC-AUC is reported next to it.

HOW THE DECISION THRESHOLD IS SUPPOSED TO WORK
----------------------------------------------
A model only outputs a PROBABILITY. Turning it into yes/no needs a threshold, and 0.5 is NOT a
good choice for an imbalanced target: `scale_pos_weight` shifts the probabilities, so 0.5 can mean
"almost never say yes". The rules used everywhere in this project:

  1. threshold-free metrics (logloss, roc_auc, average_precision) need no threshold -- they are
     computed straight from the probabilities.
  2. threshold-based metrics take an explicit `threshold`. The default 0.5 is only there so the
     function can be called without thinking; it is the "naive" reference point, not the best one.
  3. The BEST threshold is chosen on the VALIDATION set with best_threshold() and then reused on the
     test set. Never choose it on the test set -- that is leakage and inflates the score.
  4. evaluate_on_test() does points 3 for you in one call (choose on val, report on test) and also
     prints the 0.5 reference so the difference is visible.
  5. Passing threshold=None to a metric means "pick the threshold on the data I just handed you";
     that is fine for a quick look and for validation data, wrong for the test set.
"""
import numpy as np


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
def logloss(y_true, proba, eps=1e-15):
    """Binary cross-entropy = the classification LOSS."""
    p = np.clip(np.asarray(proba, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


# -----------------------------------------------------------------------------
# Threshold-based metrics
# -----------------------------------------------------------------------------
def _as_pred(y_true, proba, threshold):
    """Turn probabilities into 0/1 labels (pass threshold=0.0 if you already have hard labels)."""
    pred = (np.asarray(proba, dtype=float) >= threshold).astype(int)
    return np.asarray(y_true, dtype=int).ravel(), pred.ravel()


def _resolve_threshold(y_true, proba, threshold, metric="f1"):
    """
    threshold=None  -> pick the best one for `metric` on the data handed in (validation use);
    a number        -> use it as given (this is what the test set must always do).
    """
    if threshold is None:
        return best_threshold(y_true, proba, metric=metric)[0]
    return float(threshold)


def accuracy(y_true, proba, threshold=0.5):
    """Fraction of correct yes/no decisions. threshold=None -> best threshold for accuracy here."""
    threshold = _resolve_threshold(y_true, proba, threshold, metric="accuracy")
    y, pred = _as_pred(y_true, proba, threshold)
    return float(np.mean(pred == y))


def confusion_matrix(y_true, proba, threshold=0.5):
    """
    2x2 count matrix in scikit-learn's layout:  [[TN, FP],
                                                 [FN, TP]]
    threshold=None -> resolved with the F1-optimal threshold (there is no single "best" matrix).
    """
    threshold = _resolve_threshold(y_true, proba, threshold, metric="f1")
    y, pred = _as_pred(y_true, proba, threshold)
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tp = int(np.sum((pred == 1) & (y == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def precision(y_true, proba, threshold=0.5):
    """TP / (TP + FP) -- of everyone we flagged, how many really defaulted."""
    threshold = _resolve_threshold(y_true, proba, threshold, metric="precision")
    cm = confusion_matrix(y_true, proba, threshold)
    tp, fp = cm[1, 1], cm[0, 1]
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(y_true, proba, threshold=0.5):
    """TP / (TP + FN) -- of everyone who really defaulted, how many we caught."""
    threshold = _resolve_threshold(y_true, proba, threshold, metric="recall")
    cm = confusion_matrix(y_true, proba, threshold)
    tp, fn = cm[1, 1], cm[1, 0]
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1(y_true, proba, threshold=0.5):
    """Harmonic mean of precision and recall (2TP / (2TP + FP + FN))."""
    threshold = _resolve_threshold(y_true, proba, threshold, metric="f1")
    cm = confusion_matrix(y_true, proba, threshold)
    tp, fp, fn = cm[1, 1], cm[0, 1], cm[1, 0]
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0


# -----------------------------------------------------------------------------
# Rank-based metrics (threshold-free)
# -----------------------------------------------------------------------------
def _tie_aware_ranks(scores):
    """1-based ranks of `scores` in ascending order, ties sharing their average rank."""
    n = len(scores)
    order = np.argsort(scores, kind="stable")
    s_sorted = scores[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def roc_auc(y_true, proba):
    """
    ROC-AUC via the Mann-Whitney U form: probability that a random positive row is ranked above a
    random negative row. 0.5 = random, 1.0 = perfect. Handles tied scores.
    """
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
    """
    PR-AUC (average precision), written from scratch -- the useful metric for an
    imbalanced target: a random classifier scores the positive rate, not 0.5.
    """
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-p, kind="stable")
    y_sorted, p_sorted = y[order], p[order]
    # one curve point per DISTINCT score: rows with a tied score must be counted together,
    # otherwise the order inside the tie (arbitrary) would change the answer
    group_ends = np.append(np.where(np.diff(p_sorted) != 0)[0], len(y_sorted) - 1)
    tp = np.cumsum(y_sorted)[group_ends].astype(float)
    fp = np.cumsum(1 - y_sorted)[group_ends].astype(float)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / n_pos

    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def roc_curve_scratch(y_true, proba):
    """
    Points of the ROC curve: returns (fpr, tpr, thresholds) with one point per distinct score.
    Straight from the definition: sweep the threshold from high to low and count.
    """
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
    """
    Points of the precision-recall curve: returns (precision, recall, thresholds).
    precision[i] / recall[i] describe the prediction "score >= thresholds[i]".
    """
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


# -----------------------------------------------------------------------------
# Everything at once -- handy for the report tables and for model comparison
# -----------------------------------------------------------------------------
def classification_metrics(y_true, proba, threshold=0.5, metric="f1"):
    """
    Return one dict with loss + every threshold-free / threshold-based metric.

    threshold=0.5  -> the naive reference point (default)
    threshold=None -> the best threshold for `metric` is picked FROM THE DATA PASSED IN.
                      Use it on the validation set; for the test set pass an explicit number
                      (or just call evaluate_on_test, which does the right thing).
    The dict always reports which threshold was actually used, plus a flag telling whether it was
    auto-selected, so a report can never hide it.
    """
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
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }


def best_threshold(y_true, proba, metric="f1"):
    """
    Scan thresholds and return the one that maximises `metric` ("f1", "recall" or "accuracy").
    Select the threshold on the VALIDATION set only, then reuse it on the test set.
    """
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
    """
    The honest way to report threshold-based metrics, in one call:

      1. choose the threshold on the VALIDATION set  (never on the test set)
      2. report the test metrics at that threshold
      3. also report the 0.5 reference, so the improvement is visible

    Returns one dict, ready to become a row of the benchmark table (prefixes: val_, test_).
    """
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
