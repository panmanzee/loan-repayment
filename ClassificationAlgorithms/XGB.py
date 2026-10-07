"""
XGBoost (from scratch) - Classification task (target: not.fully.paid)

Everything is written from scratch with NumPy: the regression tree (weak learner) AND the boosting loop.

WHAT MAKES THIS "XGBoost" AND NOT PLAIN GRADIENT BOOSTING
--------------------------------------------------------
Classic gradient boosting fits each new tree to the RESIDUAL (y - p) and takes a fixed step.
XGBoost keeps the same "sum of trees" idea but adds four things, and this file implements all four:

  1. SECOND-ORDER (Newton) optimisation.  Each row contributes a gradient g_i AND a curvature h_i,
     and a leaf's value is solved analytically instead of just averaging the residuals:
         leaf value w* = - G / (H + lambda)        with G = sum(g_i), H = sum(h_i) in that leaf
  2. A SPLIT QUALITY SCORE built from those same sums (the "similarity score"):
         similarity = G^2 / (H + lambda)
         gain = 1/2 [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda) ] - gamma
     A split is only made when gain > 0, so `gamma` acts as a minimum-improvement threshold.
  3. REGULARISATION inside the objective: `reg_lambda` (L2 on leaf values, XGBoost's `lambda`) and
     `gamma` (XGBoost's `gamma` / min_split_loss), plus `min_child_weight` on H (not on row counts!).
  4. SPEED: features are PRE-BINNED once into `max_bins` quantile buckets and every split search uses
     histograms of (g, h) with cumulative sums -- this is XGBoost's `tree_method="hist"`.

THE MATH FOR BINARY:LOGISTIC (the objective used here)
------------------------------------------------------
  p_i        = sigmoid(F_i)                F = current score (log-odds), starts at base_score
  loss       = -[ y*log(p) + (1-y)*log(1-p) ]
  g_i        = dL/dF = p_i - y_i                       (first order  -> "gradient" boosting)
  h_i        = d2L/dF^2 = p_i * (1 - p_i)              (second order -> why XGBoost is "extreme")
  update     = F <- F + learning_rate * w*             (shrinkage keeps each tree small)

IMBALANCED DATA
---------------
`scale_pos_weight` multiplies the g and h of every positive row (this is exactly what XGBoost does),
so the minority class pushes harder without duplicating or re-sampling any row.

METRICS
-------
All evaluation lives in Common/metrics_classification.py (logloss, accuracy,
precision/recall/F1, confusion matrix, PR-AUC, ROC-AUC) so every classification model
added to this folder is measured in exactly the same way.

Run the built-in smoke test:   python ClassificationAlgorithms/XGB.py
(prints loss / accuracy on the real DataSet and compares the scratch model with scikit-learn.
 It only prints -- it writes no files and changes nothing else.)
"""
import os
import sys
import time

import numpy as np

# shared evaluation metrics live in Common/metrics_classification.py (one source of truth)
_COMMON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Common")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

from metrics_classification import (accuracy, average_precision, best_threshold,logloss)


# -----------------------------------------------------------------------------
# Only the link function stays local: sigmoid belongs to the OBJECTIVE (binary:logistic),
# not to evaluation. Every metric lives in Common/metrics_classification.py.
# -----------------------------------------------------------------------------
def sigmoid(F):
    """Stable 1 / (1 + exp(-F))."""
    F = np.clip(F, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-F))


def sqrt_scale_pos_weight(y):
    """
    The class-imbalance weight used throughout this project (same two lines as ML-project/final.ipynb):

        raw_weight           = (n_rows - n_positive) / n_positive      # how many negatives per positive
        scale_pos_weight     = sqrt(raw_weight)

    The square root keeps the minority class pushing harder without letting it dominate the loss
    (the raw ratio is far too aggressive). It is computed from the labels handed to fit(), i.e. from
    the TRAINING rows only -- never from the validation/test rows.
    """
    y = np.asarray(y, dtype=int).ravel()
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return 1.0
    raw_weight = (len(y) - n_pos) / n_pos
    return float(np.sqrt(raw_weight))


# -----------------------------------------------------------------------------
# The weak learner: one CART regression tree grown on (g, h)
# -----------------------------------------------------------------------------
class XGBTreeScratch:
    """
    A small regression tree that splits on the histogram of g and h.

    At every node it tries every allowed feature and every bin boundary, and keeps the split with the
    largest `gain` (the XGBoost formula). A node becomes a leaf when max_depth is reached, when it is
    too small (min_child_weight / min_samples_leaf) or when no split improves the score by more than
    `gamma`. A leaf predicts -G/(H+lambda).

    Inputs are already-binned features `Xb` (integers), so all internal work is bincount + cumsum.
    """

    def __init__(self, max_depth=3, reg_lambda=1.0, gamma=0.0, min_child_weight=1.0,
                 min_samples_leaf=1, max_bins=256):
        self.max_depth = int(max_depth)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.min_child_weight = float(min_child_weight)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_bins = int(max_bins)

    # -- tree structure -------------------------------------------------------
    def _new_node(self):
        self.feature.append(-1)          # -1 = leaf
        self.threshold_bin.append(0)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(0.0)           # leaf value (-G/(H+lambda))
        self.cover.append(0.0)           # sum of h in that node (XGBoost calls it "cover")
        self.n_rows.append(0)
        return len(self.value) - 1

    def _best_split(self, Xb, g, h, rows, G, H, feature_idx):
        """Return (feature, bin, gain) of the best split, or None when nothing beats gamma."""
        parent_term = G * G / (H + self.reg_lambda)
        n_rows = len(rows)
        best = None
        best_gain = 0.0

        for j in feature_idx:
            b = Xb[rows, j]
            # histogram of gradients / hessians per bin
            g_hist = np.bincount(b, weights=g[rows], minlength=self.max_bins)
            h_hist = np.bincount(b, weights=h[rows], minlength=self.max_bins)
            c_hist = np.bincount(b, minlength=self.max_bins)

            # every bin boundary k = "go left if bin <= k"
            G_L = np.cumsum(g_hist)[:-1]
            H_L = np.cumsum(h_hist)[:-1]
            R_L = np.cumsum(c_hist)[:-1]
            G_R = G - G_L
            H_R = H - H_L
            R_R = n_rows - R_L

            valid = ((H_L >= self.min_child_weight) & (H_R >= self.min_child_weight)
                     & (R_L >= self.min_samples_leaf) & (R_R >= self.min_samples_leaf))
            if not valid.any():
                continue

            score = (G_L * G_L / (H_L + self.reg_lambda)
                     + G_R * G_R / (H_R + self.reg_lambda)
                     - parent_term)
            score = np.where(valid, score, -np.inf)
            k = int(np.argmax(score))
            gain = 0.5 * score[k] - self.gamma
            if gain > best_gain:
                best_gain = gain
                best = (int(j), k, float(gain))

        return best

    def fit(self, Xb, g, h, rows, feature_idx):
        """Grow the tree on `rows` only (row subsampling) and on `feature_idx` only (column subsampling)."""
        self.n_features_ = Xb.shape[1]
        self.feature, self.threshold_bin, self.left, self.right = [], [], [], []
        self.value, self.cover, self.n_rows = [], [], []
        self.gain_per_feature_ = np.zeros(self.n_features_)
        self.total_gain_ = 0.0

        root = self._new_node()
        stack = [(np.asarray(rows, dtype=int), 0, root)]

        while stack:
            node_rows, depth, node = stack.pop()
            G = float(g[node_rows].sum())
            H = float(h[node_rows].sum())
            self.value[node] = -G / (H + self.reg_lambda)     # XGBoost leaf weight
            self.cover[node] = H
            self.n_rows[node] = len(node_rows)

            if depth >= self.max_depth:
                continue
            if H < 2.0 * self.min_child_weight:
                continue
            if len(node_rows) < 2 * self.min_samples_leaf:
                continue

            split = self._best_split(Xb, g, h, node_rows, G, H, feature_idx)
            if split is None:
                continue

            j, k, gain = split
            self.feature[node] = j
            self.threshold_bin[node] = k
            self.gain_per_feature_[j] += gain
            self.total_gain_ += gain

            go_left = Xb[node_rows, j] <= k
            left_rows = node_rows[go_left]
            right_rows = node_rows[~go_left]

            left_child = self._new_node()
            right_child = self._new_node()
            self.left[node] = left_child
            self.right[node] = right_child

            stack.append((left_rows, depth + 1, left_child))
            stack.append((right_rows, depth + 1, right_child))

        self.feature = np.asarray(self.feature)
        self.threshold_bin = np.asarray(self.threshold_bin)
        self.left = np.asarray(self.left)
        self.right = np.asarray(self.right)
        self.value = np.asarray(self.value)
        self.cover = np.asarray(self.cover)
        self.n_rows = np.asarray(self.n_rows)
        self.n_nodes_ = len(self.value)
        return self

    def predict(self, Xb):
        """Leaf value for every row (vectorised traversal)."""
        node = np.zeros(Xb.shape[0], dtype=int)
        while True:
            feat = self.feature[node]
            internal = feat >= 0
            if not internal.any():
                break
            r = np.where(internal)[0]
            go_left = Xb[r, feat[r]] <= self.threshold_bin[node[r]]
            node[r] = np.where(go_left, self.left[node[r]], self.right[node[r]])
        return self.value[node]


# -----------------------------------------------------------------------------
# The ensemble
# -----------------------------------------------------------------------------
class XGBClassifierScratch:
    """
    Binary XGBoost classifier written from scratch (logistic objective, histogram splits).

    Parameters (same names as the real xgboost library)
    ---------------------------------------------------
    n_estimators      number of boosting rounds (trees)                -> key "better model" knob
    learning_rate     shrinkage applied to every tree (eta)            -> smaller + more trees
    max_depth         depth of each weak learner                       -> how complex one tree may be
    reg_lambda        L2 penalty on leaf values (lambda)              -> shrinks leaf weights
    gamma             minimum gain required to make a split            -> prunes useless branches
    min_child_weight  minimum sum of h in a child (NOT a row count)     -> guards against thin leaves
    subsample         fraction of ROWS used per tree                   -> stochastic boosting
    colsample_bytree  fraction of FEATURES tried per tree              -> decorrelates the trees
    scale_pos_weight  weight applied to positive rows (imbalance knob).
                      "auto" (default) = sqrt((n - n_pos) / n_pos) from the TRAINING labels --
                      the same formula used in ML-project/final.ipynb (differs from xgboost, whose
                      default is 1.0). Pass a number to override, or 1.0 to switch it off.
    max_bins          number of quantile buckets used for the histograms
    base_score        starting log-odds (default = log-odds of the weighted positive rate)

    Usage
    -----
        model = XGBClassifierScratch(n_estimators=300, max_depth=3, learning_rate=0.1)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30)
        proba = model.predict_proba(X_test)[:, 1]      # same shape as scikit-learn
        pred  = model.predict(X_test, threshold=model.best_threshold_)
    """

    def __init__(self, n_estimators=300, learning_rate=0.1, max_depth=3, reg_lambda=1.0,
                 gamma=0.0, min_child_weight=1.0, min_samples_leaf=1, subsample=1.0,
                 colsample_bytree=1.0, scale_pos_weight="auto", max_bins=256,
                 base_score=None, random_state=42):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.min_child_weight = float(min_child_weight)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        if isinstance(scale_pos_weight, str) and scale_pos_weight.lower() != "auto":
            raise ValueError('scale_pos_weight must be "auto" or a number')
        self.scale_pos_weight = scale_pos_weight
        self.max_bins = int(max_bins)
        self.base_score = base_score
        self.random_state = random_state

    # -- binning --------------------------------------------------------------
    def _make_bin_edges(self, column):
        """Quantile cut points of one feature (duplicates removed -> 'the bins' XGBoost uses)."""
        probs = np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]
        edges = np.unique(np.quantile(column, probs))
        return edges

    def _bin(self, X):
        """Turn raw features into bin indices with the edges learned on the training data."""
        Xb = np.empty(X.shape, dtype=np.int32)
        for j in range(X.shape[1]):
            Xb[:, j] = np.digitize(X[:, j], self.bin_edges_[j])
        return Xb

    # -- fit ------------------------------------------------------------------
    def fit(self, X, y, eval_set=None, feature_names=None, early_stopping_rounds=None,
            eval_metric="logloss", verbose=False):
        """
        Train the ensemble.

        eval_set              list like [(X_val, y_val)] -- used for the loss curve and early stopping
        early_stopping_rounds stop when the eval metric has not improved for that many rounds
                              (the model keeps the best round, like xgboost does)
        eval_metric           "logloss" or "aucpr"
        """
        t0 = time.time()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"f{j}" for j in range(p)]

        rng = np.random.RandomState(self.random_state)

        # 1. pre-bin every feature once (this is what makes the split search cheap)
        self.bin_edges_ = [self._make_bin_edges(X[:, j]) for j in range(p)]
        Xb = self._bin(X)
        eval_binned = None
        if eval_set is not None:
            eval_binned = [(self._bin(np.asarray(Xe, dtype=float)), np.asarray(ye, dtype=int))
                           for Xe, ye in eval_set]

        # 2. per-row weights: positives get scale_pos_weight (XGBoost multiplies g and h by it)
        if isinstance(self.scale_pos_weight, str) and self.scale_pos_weight.lower() == "auto":
            self.scale_pos_weight_ = sqrt_scale_pos_weight(y)     # <- computed from y_train
        else:
            self.scale_pos_weight_ = float(self.scale_pos_weight)
        w = np.where(y == 1, self.scale_pos_weight_, 1.0).astype(float)

        # 3. starting score = log-odds of the (weighted) positive rate
        if self.base_score is None:
            p0 = float(np.sum(w * y) / np.sum(w))
            p0 = min(max(p0, 1e-6), 1.0 - 1e-6)
            self.base_score_ = float(np.log(p0 / (1.0 - p0)))
        else:
            self.base_score_ = float(self.base_score)

        F = np.full(n, self.base_score_)
        self.trees = []
        self.train_loss = []
        self.evals_result_ = {"validation": {eval_metric: []}} if eval_binned else {}
        self.best_iteration_ = self.n_estimators
        self.best_score_ = None

        n_rows_sub = max(1, int(round(self.subsample * n)))
        n_cols_sub = max(1, int(round(self.colsample_bytree * p)))

        for m in range(self.n_estimators):
            # 4. gradient and hessian of the logistic loss at the current prediction
            proba = sigmoid(F)
            g = w * (proba - y)
            h = w * proba * (1.0 - proba)
            h = np.maximum(h, 1e-12)                      

            # 5. row + column subsampling for this tree
            rows = (rng.choice(n, size=n_rows_sub, replace=False) if self.subsample < 1.0
                    else np.arange(n))
            features = (rng.choice(p, size=n_cols_sub, replace=False) if self.colsample_bytree < 1.0
                        else np.arange(p))

            tree = XGBTreeScratch(max_depth=self.max_depth, reg_lambda=self.reg_lambda,
                                 gamma=self.gamma, min_child_weight=self.min_child_weight,
                                 min_samples_leaf=self.min_samples_leaf, max_bins=self.max_bins)
            tree.fit(Xb, g, h, rows, features)
            self.trees.append(tree)

            # 6. shrinkage step: only the rows that reach a leaf get its value added
            F = F + self.learning_rate * tree.predict(Xb)
            self.train_loss.append(logloss(y, sigmoid(F)))

            # 7. evaluation / early stopping
            if eval_binned:
                Xe, ye = eval_binned[0]
                Fe = np.full(Xe.shape[0], self.base_score_)
                for t in self.trees:
                    Fe = Fe + self.learning_rate * t.predict(Xe)
                score = (logloss(ye, sigmoid(Fe)) if eval_metric == "logloss"
                         else -average_precision(ye, sigmoid(Fe)))
                self.evals_result_["validation"][eval_metric].append(float(score))
                if self.best_score_ is None or score < self.best_score_ - 1e-12:
                    self.best_score_ = score
                    self.best_iteration_ = m + 1
                elif (early_stopping_rounds is not None
                      and (m + 1) - self.best_iteration_ >= early_stopping_rounds):
                    if verbose:
                        print(f"  early stopping at round {m + 1} "
                              f"(best round {self.best_iteration_}, {eval_metric}={self.best_score_:.5f})")
                    break
            if verbose and (m + 1) % 25 == 0:
                print(f"  round {m + 1:>4} | train logloss {self.train_loss[-1]:.5f}")

        self.n_estimators_ = len(self.trees)
        self.early_stopping_used_ = bool(eval_binned) and early_stopping_rounds is not None
        self.fit_time_ = time.time() - t0
        # importance = share of the total gain each feature produced (XGBoost's "gain" importance)
        gain = np.sum([t.gain_per_feature_ for t in self.trees], axis=0)
        self.feature_importances_ = gain / gain.sum() if gain.sum() > 0 else gain
        return self

    # -- prediction -----------------------------------------------------------
    def decision_function(self, X, iteration_limit=None):
        """
        Raw score F (log-odds) for each row.

        When early stopping was used, the default prediction uses the trees up to
        `best_iteration_` -- exactly what xgboost does after early stopping. Pass
        `iteration_limit=self.n_estimators_` to force every tree, or any other number.
        """
        if iteration_limit is None and getattr(self, "early_stopping_used_", False):
            iteration_limit = self.best_iteration_
        Xb = self._bin(np.asarray(X, dtype=float))
        trees = self.trees if iteration_limit is None else self.trees[:iteration_limit]
        F = np.full(Xb.shape[0], self.base_score_)
        for tree in trees:
            F = F + self.learning_rate * tree.predict(Xb)
        return F

    def predict_proba(self, X, iteration_limit=None):
        """[P(y=0), P(y=1)] -- same layout as scikit-learn so it can be swapped in."""
        p1 = sigmoid(self.decision_function(X, iteration_limit))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, threshold=0.5):
        """0/1 prediction.  Pass model.best_threshold_ to use the threshold picked on validation."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def staged_decision_function(self, X):
        """Yield the score after 1, 2, 3, ... trees (used for the performance curve)."""
        Xb = self._bin(np.asarray(X, dtype=float))
        F = np.full(Xb.shape[0], self.base_score_)
        for tree in self.trees:
            F = F + self.learning_rate * tree.predict(Xb)
            yield F.copy()

    def staged_predict_proba(self, X):
        for F in self.staged_decision_function(X):
            yield sigmoid(F)

    # -- helpers used by the report -------------------------------------------
    def feature_importance_table(self, top=None, kind="gain"):
        """List of (feature name, importance) sorted from most to least important."""
        values = self.feature_importances_ if kind == "gain" else np.array(
            [float(np.sum([t.cover[t.feature == j].sum() for t in self.trees])) for j in range(self.n_features_in_)]
        )
        order = np.argsort(-values)
        if top is not None:
            order = order[:top]
        return [(self.feature_names_[j], float(values[j])) for j in order]

    def pick_threshold(self, X_val, y_val, metric="f1"):
        """
        Choose the decision threshold on the VALIDATION set (never on the test set);
        delegates to the shared scan in Common/metrics_classification.py and stores the result.
        """
        proba = self.predict_proba(X_val)[:, 1]
        best_thr, best_score = best_threshold(y_val, proba, metric=metric)
        self.best_threshold_ = best_thr
        self.best_threshold_score_ = best_score
        return best_thr

