"""
Classification evaluation shared by every classification notebook.

The flagging threshold is NOT learned by the models: it is chosen on the VALIDATION set by a rule
(FLAG_RULE, F1.5 by default = recall weighs 1.5x precision, because missing a defaulter costs more
than reviewing a good payer). The test set is only used to report the numbers at that threshold.
"""

import re

import numpy as np

from metrics_classification import accuracy, classification_metrics, f1, precision, recall

FLAG_RULE = "f1.5"
TOPK = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BETTER = "Blend EBM+MLP (better model)"

# one fixed colour per model, used in every chart
COLORS = {
    "Logistic Regression": "#2a78d6",
    "Decision Tree": "#eb6834",
    "Random Forest": "#1baf7a",
    "Gradient Boosting": "#eda100",
    "XGBoost": "#8b5cf6",
    "SVC (linear)": "#00a3a3",
    "SVC (rbf)": "#00838f",
    "k-NN": "#c2185b",
    "Naive Bayes": "#d4549a",
    "Perceptron (averaged)": "#0f9bb3",
    "Single-Layer Perceptron": "#9aa31a",
    "MLP": "#8d6e63",
    "EBM (additive)": "#111111",
    "Stacking (5 bases)": "#7b1fa2",
    BETTER: "#d62728",
}


def _counts(y_true, proba, thr):
    """tp, fp, fn, tn at one threshold."""
    y = np.asarray(y_true, dtype=int).ravel()
    pred = (np.asarray(proba, dtype=float).ravel() >= thr).astype(int)
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    tn = float(np.sum((pred == 0) & (y == 0)))
    return tp, fp, fn, tn


def f_beta(y_true, proba, thr, beta=2.0):
    """F-beta score at one threshold. beta=1 -> F1, beta=2 -> recall counts twice as much as precision."""
    tp, fp, fn, _ = _counts(y_true, proba, thr)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec) if prec + rec else 0.0


def youden_j(y_true, proba, thr):
    """sensitivity + specificity - 1 (the ROC point furthest above the diagonal)."""
    tp, fp, fn, tn = _counts(y_true, proba, thr)
    return (tp / (tp + fn) if tp + fn else 0.0) + (tn / (tn + fp) if tn + fp else 0.0) - 1.0


def choose_threshold(y_true, proba, rule=FLAG_RULE, base_rate=None):
    """Scan the thresholds and return the one that maximises `rule`, using the VALIDATION scores only.

    "f<beta>"              maximum F-beta: "f1" (balanced), "f1.5" (headline rule), "f2" (recall counts twice)
    "precision>=<k>x base" highest recall whose precision stays at least k x the base rate
    "youden"               sensitivity + specificity - 1, the classic ROC-based rule
    "accuracy"             the naive reference (answers almost everybody 'pays' on 16% positives)
    """
    p = np.asarray(proba, dtype=float).ravel()
    grid = np.unique(np.concatenate([np.linspace(0.001, 0.999, 499), p]))
    base = float(np.mean(y_true)) if base_rate is None else float(base_rate)
    m_fl = re.match(r"precision>=\s*([0-9.]+)\s*x", rule)
    floor = float(m_fl.group(1)) * base if m_fl else None
    m_fb = re.match(r"f([0-9.]+)$", rule)
    beta = float(m_fb.group(1)) if m_fb else None
    best_thr, best_score = 0.5, -np.inf
    for thr in grid:
        if floor is not None:
            score = recall(y_true, p, thr) if precision(y_true, p, thr) >= floor else -1.0
        elif beta is not None:
            score = f_beta(y_true, p, thr, beta)
        elif rule == "youden":
            score = youden_j(y_true, p, thr)
        else:
            score = accuracy(y_true, p, thr)
        if score > best_score:
            best_score, best_thr = float(score), float(thr)
    return best_thr, best_score


def evaluate_model(y_val, p_val, y_test, p_test, rule=FLAG_RULE, base_rate=None):
    """One honest report row:
    1. choose the flagging threshold on the VALIDATION set (rule = FLAG_RULE)
    2. report every test metric at that threshold
    3. also report the plain 0.5 and the F1 / Youden rules, so the trade-off stays visible
    """
    thr, val_score = choose_threshold(y_val, p_val, rule=rule, base_rate=base_rate)
    row = {"chosen_threshold": float(thr), "val_" + rule: float(val_score), "threshold_rule": rule}
    cm = classification_metrics(y_test, p_test, threshold=thr)
    for k in ("logloss", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"):
        row["test_" + k] = cm[k]
    for k, fn_ in (("accuracy", accuracy), ("precision", precision), ("recall", recall), ("f1", f1)):
        row[f"test_{k}@0.5"] = float(fn_(y_test, p_test, 0.5))
    thr_f1, _ = choose_threshold(y_val, p_val, rule="f1")
    row["thr_f1_rule"] = float(thr_f1)
    row["test_recall@f1_rule"] = float(recall(y_test, p_test, thr_f1))
    row["test_precision@f1_rule"] = float(precision(y_test, p_test, thr_f1))
    thr_yj, _ = choose_threshold(y_val, p_val, rule="youden")
    row["thr_youden_rule"] = float(thr_yj)
    row["test_recall@youden_rule"] = float(recall(y_test, p_test, thr_yj))
    row["test_precision@youden_rule"] = float(precision(y_test, p_test, thr_yj))
    row["test_specificity"] = row["test_tn"] / (row["test_tn"] + row["test_fp"])
    row["test_balanced_accuracy"] = (row["test_recall"] + row["test_specificity"]) / 2
    return row


def topk_table(y_test, p_test, fracs=TOPK):
    """Rank the applicants by predicted risk and flag the riskiest x%: how many defaulters does that catch?"""
    y = np.asarray(y_test, dtype=int).ravel()
    p = np.asarray(p_test, dtype=float).ravel()
    order = np.argsort(-p)
    n, n_pos = len(y), int(y.sum())
    rows = []
    for f in fracs:
        k = max(1, int(round(f * n)))
        caught = int(y[order[:k]].sum())
        prec = caught / k
        rows.append(
            {
                "flagged share": f,
                "applicants flagged": k,
                "defaulters caught": caught,
                "recall (defaulters caught / all)": caught / n_pos,
                "precision inside the flagged group": prec,
                "lift vs flagging at random": prec / (n_pos / n),
            }
        )
    return rows


def fit_model(model, use_val, X_tr, y_tr, X_va, y_va, names):
    """Fit one model; models that use the validation set get it for early stopping."""
    if use_val:
        return model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], feature_names=names, early_stopping_rounds=30)
    return model.fit(X_tr, y_tr, feature_names=names)


def staged_pos(model, X):
    """P(class 1) after every training stage (boosting round / gradient step)."""
    out = []
    for p in model.staged_predict_proba(X):
        a = np.asarray(p, dtype=float)
        out.append(a[:, 1] if a.ndim == 2 else a.ravel())
    return out


def stratified_folds(y, n_splits=5, seed=42):
    """k folds with (almost) the same positive rate in each. Returns [(train_idx, test_idx), ...]."""
    rng = np.random.RandomState(seed)
    buckets = [[] for _ in range(n_splits)]
    for cls in (0, 1):
        idx = np.where(np.asarray(y) == cls)[0]
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            buckets[i % n_splits].append(j)  # deal the shuffled class members round-robin
    folds = []
    for f in buckets:
        te = np.sort(np.asarray(f, dtype=int))
        folds.append((np.setdiff1d(np.arange(len(y)), te), te))
    return folds
