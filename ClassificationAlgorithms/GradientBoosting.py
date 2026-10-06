"""
Gradient Boosting (from scratch) - Classification task (target: not.fully.paid)

Everything is written from scratch with NumPy: the regression tree (weak learner) AND the boosting loop.
Data comes from Common/data_classification.py, every number is measured with Common/metrics_classification.py.

GRADIENT BOOSTING (Friedman, 1999-2001) IN ONE PICTURE
------------------------------------------------------
One decision tree alone is a rough guess. Boosting makes a strong model by adding many SMALL trees,
where every new tree is fitted to what the current ensemble still gets WRONG:

    F_0(x) = log-odds of the positive rate                 start: one number for everybody
    for m = 1 .. M:
        p_i = sigmoid(F_{m-1}(x_i))                        current probability
        r_i = (y_i - p_i) / (p_i * (1 - p_i))              PSEUDO-RESIDUAL = how wrong we still are
        pick a random `subsample` fraction of the rows      stochastic gradient boosting
        h_m  = small regression tree fitted to (x_i, r_i)   weak learner: splits by VARIANCE REDUCTION
        v_j  = value of leaf j, found by a line search      one Newton step on that leaf's rows
        F_m(x) = F_{m-1}(x) + learning_rate * v_j           take a small step, never a leap

Why "gradient" boosting: for a loss L, the negative gradient -dL/dF is exactly the direction that
lowers the loss fastest. For logistic loss that direction is proportional to (y - p), so fitting the
tree to residuals IS gradient descent in function space - the tree just chooses a good step size per leaf.

HOW THIS DIFFERS FROM XGBoost (ClassificationAlgorithms/XGB.py)
--------------------------------------------------------------
Both build "a sum of small trees", but the two algorithms do three things differently, and this file
keeps the CLASSIC GB choices on purpose - that is the point of having both models:

                              Gradient Boosting (this file)          XGBoost (XGB.py)
    what the tree learns      pseudo-residual r (1st order only)      gradients g AND hessians h (2nd order)
    split criterion           variance reduction of r                similarity gain with lambda
    leaf value                line search: -sum(w g)/sum(w h)         closed form: -G/(H+lambda)
    regularisation            none built in (depth + lr + sub-    lambda, gamma (min split gain),
                              sample are the only brakes)             min_child_weight on H
    class weighting           sample weights w (also inside the       scale_pos_weight on g and h
                              split search)

The weight convention is the project one (same as ML-project/final.ipynb):
    raw = (n - n_pos) / n_pos,   scale_pos_weight = sqrt(raw)   -> see sqrt_scale_pos_weight().

Metrics and the decision threshold
----------------------------------
logloss / aucpr come from Common/metrics_classification.py. `eval_metric` only decides WHEN to stop and
what is recorded in `evals_result_`; the model always learns the logistic loss. The prediction threshold
is chosen on the VALIDATION set (pick_threshold -> metrics_classification.best_threshold) and then reused
on the test set - never chosen on the test set.

Example
-------
    import sys; sys.path.insert(0, "Common")
    from data_classification import build_classification_data, load_dataset, train_val_test_split_scratch
    from metrics_classification import evaluate_on_test

    X, y, names = build_classification_data(load_dataset())
    X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test_split_scratch(X, y)

    gb = GradientBoostingClassifierScratch(n_estimators=300, learning_rate=0.05, max_depth=3)
    gb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], feature_names=names, early_stopping_rounds=30)
    report = evaluate_on_test(y_va, gb.predict_proba(X_va)[:, 1], y_te, gb.predict_proba(X_te)[:, 1])
"""
import os
import sys
import time

import numpy as np

# -----------------------------------------------------------------------------
# Shared helpers: metrics come from Common/metrics_classification.py, nothing is duplicated from there.
# -----------------------------------------------------------------------------
_COMMON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Common")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

from metrics_classification import (accuracy, average_precision, best_threshold,logloss)

# sigmoid belongs to the OBJECTIVE (binary:logistic), not to evaluation -> it stays with the model.
def sigmoid(F):
    """Stable 1 / (1 + exp(-F))."""
    F = np.clip(F, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-F))


def sqrt_scale_pos_weight(y):
    """
    The class-imbalance weight used throughout this project (same two lines as ML-project/final.ipynb):

        raw_weight       = (n_rows - n_positive) / n_positive
        scale_pos_weight = sqrt(raw_weight)

    In this file the weight is used as a SAMPLE WEIGHT w_i: positives get scale_pos_weight, negatives 1.
    It then appears in three places -- the split search (weighted variance), the leaf line search
    (weighted Newton step) and the base score -- which is the classic-GB way of handling imbalance.
    """
    y = np.asarray(y, dtype=int).ravel()
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return 1.0
    return float(np.sqrt((len(y) - n_pos) / n_pos))


# -----------------------------------------------------------------------------
# Weak learner: one CART regression tree fitted to the pseudo-residuals
# -----------------------------------------------------------------------------
class GBTreeScratch:
    """
    A small regression tree whose splits minimise the weighted squared error of the pseudo-residuals.

    Only the tree STRUCTURE is decided here (which feature, which cut). The leaf VALUES are filled in
    afterwards by the booster with a one-step Newton line search, because that value depends on the loss.

    Weighted error of a node = sum(w_i * (r_i - r_bar)^2) = sum(w r^2) - (sum(w r))^2 / sum(w)
    so the reduction of that error for a split is
        gain = (A_L^2 / W_L) + (A_R^2 / W_R) - (A^2 / W),     A = sum(w r),  W = sum(w)
    (the sum(w r^2) parts cancel). The tree keeps the cut with the largest gain.

    Implemented with a single pre-binning of the features (quantile buckets) so the split search is a
    bincount + cumsum per feature instead of a full sort of every node.
    """

    def __init__(self, max_depth=3, min_samples_leaf=5, max_bins=256):
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_bins = int(max_bins)

    # -- tree structure -------------------------------------------------------
    def _new_node(self):
        self.feature.append(-1)          # -1 = leaf
        self.threshold_bin.append(0)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(0.0)           # filled in by the booster (line search)
        self.n_rows.append(0)
        return len(self.value) - 1

    def build(self, Xb, r, w, rows, feature_idx):
        """Grow the tree on `rows` (row subsampling) using only `feature_idx` (column subsampling)."""
        self.n_features_ = Xb.shape[1]
        self.feature, self.threshold_bin, self.left, self.right = [], [], [], []
        self.value, self.n_rows = [], []
        self.gain_per_feature_ = np.zeros(self.n_features_)
        self.total_gain_ = 0.0

        A = w * r                       # what the split search actually needs
        W = w
        Xb_sub = Xb[:, feature_idx]      # column subsampling: the search only sees these features

        root = self._new_node()
        stack = [(np.asarray(rows, dtype=int), 0, root)]
        while stack:
            node_rows, depth, node = stack.pop()
            self.n_rows[node] = len(node_rows)

            if depth >= self.max_depth or len(node_rows) < 2 * self.min_samples_leaf:
                continue

            total_a = float(A[node_rows].sum())
            total_w = float(W[node_rows].sum())
            if total_w <= 2e-12:
                continue

            split = self._best_split(Xb_sub, A, W, node_rows, total_a, total_w, feature_idx)
            if split is None:
                continue

            j, k, gain = split
            self.feature[node] = j
            self.threshold_bin[node] = k
            self.gain_per_feature_[j] += gain
            self.total_gain_ += gain

            go_left = Xb[node_rows, j] <= k
            left_child = self._new_node()
            right_child = self._new_node()
            self.left[node] = left_child
            self.right[node] = right_child
            stack.append((node_rows[go_left], depth + 1, left_child))
            stack.append((node_rows[~go_left], depth + 1, right_child))

        for name in ("feature", "threshold_bin", "left", "right", "value", "n_rows"):
            setattr(self, name, np.asarray(getattr(self, name)))
        self.n_nodes_ = len(self.value)
        return self

    def _best_split(self, Xb_sub, A, W, rows, total_a, total_w, feature_idx):
        """Weighted variance-reduction search on the subset of columns offered to this tree."""
        best, best_gain = None, 0.0
        n_rows = len(rows)
        parent_term = total_a * total_a / total_w
        for local_j in range(Xb_sub.shape[1]):
            b = Xb_sub[rows, local_j]
            a_hist = np.bincount(b, weights=A[rows], minlength=self.max_bins)
            w_hist = np.bincount(b, weights=W[rows], minlength=self.max_bins)
            n_hist = np.bincount(b, minlength=self.max_bins)

            A_L = np.cumsum(a_hist)[:-1]
            W_L = np.cumsum(w_hist)[:-1]
            N_L = np.cumsum(n_hist)[:-1]
            A_R = total_a - A_L
            W_R = total_w - W_L
            N_R = n_rows - N_L

            valid = ((N_L >= self.min_samples_leaf) & (N_R >= self.min_samples_leaf)
                     & (W_L > 1e-12) & (W_R > 1e-12))
            if not valid.any():
                continue
            score = (A_L * A_L / np.where(W_L > 1e-12, W_L, 1.0)
                     + A_R * A_R / np.where(W_R > 1e-12, W_R, 1.0)
                     - parent_term)
            score = np.where(valid, score, -np.inf)
            k = int(np.argmax(score))
            if score[k] > best_gain:
                best_gain = float(score[k])
                best = (int(feature_idx[local_j]), k, float(score[k]))
        return best

    def apply(self, Xb):
        """Leaf index of every row (vectorised traversal)."""
        node = np.zeros(Xb.shape[0], dtype=int)
        while True:
            feat = self.feature[node]
            internal = feat >= 0
            if not internal.any():
                break
            r = np.where(internal)[0]
            go_left = Xb[r, feat[r]] <= self.threshold_bin[node[r]]
            node[r] = np.where(go_left, self.left[node[r]], self.right[node[r]])
        return node


# -----------------------------------------------------------------------------
# The ensemble
# -----------------------------------------------------------------------------
class GradientBoostingClassifierScratch:
    """
    Binary Gradient Boosting classifier written from scratch (logistic loss, first-order boosting).

    Parameters
    ----------
    n_estimators      number of boosting rounds (trees)
    learning_rate     shrinkage of every leaf value (the "slow learning" knob)
    max_depth         depth of each weak learner
    min_samples_leaf  minimum ROWS in a leaf (classic GB brake; note XGBoost uses min_child_weight).
                      Default 5 -- scikit-learn's GB default is 1, pass 1 for a like-for-like run
    subsample         fraction of ROWS per tree (Friedman's stochastic gradient boosting; < 1 fights overfitting)
    max_features      fraction of FEATURES offered to each tree (like sklearn's max_features)
    leaf_solver       "newton" (default) = one Newton step per leaf, -sum(w g)/sum(w h)
                      "mean"             = plain average of the pseudo-residuals (a pure gradient step)
    scale_pos_weight  weight of the positive rows: "auto" (default) = sqrt((n - n_pos) / n_pos) from y_train,
                      or any number, or 1.0 to switch the weighting off completely
    max_bins          quantile buckets used for the split search
    base_score        starting log-odds (default = log-odds of the weighted positive rate)
    random_state      seed for row/feature subsampling

    Fitted attributes
    -----------------
    trees, train_loss, evals_result_, best_iteration_, n_estimators_, early_stopping_used_,
    scale_pos_weight_, feature_importances_, feature_names_, bin_edges_, fit_time_
    """

    def __init__(self, n_estimators=300, learning_rate=0.1, max_depth=3, min_samples_leaf=5,
                 subsample=1.0, max_features=1.0, leaf_solver="newton", scale_pos_weight="auto",
                 max_bins=256, base_score=None, random_state=42):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.max_features = float(max_features)
        if leaf_solver not in ("newton", "mean"):
            raise ValueError('leaf_solver must be "newton" or "mean"')
        self.leaf_solver = leaf_solver
        if isinstance(scale_pos_weight, str) and scale_pos_weight.lower() != "auto":
            raise ValueError('scale_pos_weight must be "auto" or a number')
        self.scale_pos_weight = scale_pos_weight
        self.max_bins = int(max_bins)
        self.base_score = base_score
        self.random_state = random_state

    # -- binning --------------------------------------------------------------
    def _make_bin_edges(self, column):
        """Quantile cut points of one feature (duplicates removed)."""
        probs = np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]
        return np.unique(np.quantile(column, probs))

    def _bin(self, X):
        Xb = np.empty(X.shape, dtype=np.int32)
        for j in range(X.shape[1]):
            Xb[:, j] = np.digitize(X[:, j], self.bin_edges_[j])
        return Xb

    # -- fit ------------------------------------------------------------------
    def fit(self, X, y, eval_set=None, feature_names=None, early_stopping_rounds=None,
            eval_metric="logloss", verbose=False):
        """
        Train the ensemble.  `eval_set` is a list like [(X_val, y_val)]; the first entry is used for the
        loss curve and for early stopping (the booster keeps `best_iteration_`, like sklearn/xgboost).
        `eval_metric` is "logloss" or "aucpr" and only affects WHEN training stops (see the module docstring).
        """
        t0 = time.time()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"f{j}" for j in range(p)]

        rng = np.random.RandomState(self.random_state)

        # 1. bin the features once (speed only -- the model sees the same information)
        self.bin_edges_ = [self._make_bin_edges(X[:, j]) for j in range(p)]
        Xb = self._bin(X)
        eval_binned = None
        if eval_set is not None:
            eval_binned = [(self._bin(np.asarray(Xe, dtype=float)), np.asarray(ye, dtype=int))
                           for Xe, ye in eval_set]

        # 2. sample weights: positives get scale_pos_weight (classic GB uses weights, not resampling)
        if isinstance(self.scale_pos_weight, str) and self.scale_pos_weight.lower() == "auto":
            self.scale_pos_weight_ = sqrt_scale_pos_weight(y)
        else:
            self.scale_pos_weight_ = float(self.scale_pos_weight)
        w = np.where(y == 1, self.scale_pos_weight_, 1.0).astype(float)

        # 3. starting score F_0 = log-odds of the (weighted) positive rate
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
        n_cols_sub = max(1, int(round(self.max_features * p)))

        for m in range(self.n_estimators):
            # 4. pseudo-residuals of the logistic loss:  r = (y - p) / (p (1 - p))
            proba = sigmoid(F)
            h = np.maximum(proba * (1.0 - proba), 1e-12)      # curvature, also the Newton denominator
            g = proba - y                                     # dL/dF
            r_pseudo = -g / h                                 # = (y - p) / (p (1 - p))

            rows = (rng.choice(n, size=n_rows_sub, replace=False) if self.subsample < 1.0
                    else np.arange(n))
            features = (rng.choice(p, size=n_cols_sub, replace=False) if self.max_features < 1.0
                        else np.arange(p))

            # 5. weak learner: regression tree on the pseudo-residuals (variance reduction)
            tree = GBTreeScratch(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                 max_bins=self.max_bins)
            tree.build(Xb, r_pseudo, w, rows, features)

            # 6. line search per leaf -> the ADDITIVE update  F += learning_rate * v_leaf
            leaf = tree.apply(Xb)
            order = np.argsort(leaf, kind="stable")
            leaf_sorted, row_order = leaf[order], order
            bounds = np.searchsorted(leaf_sorted, np.arange(tree.n_nodes_ + 1))   # n_nodes_ + 1 edges
            for node_id in range(tree.n_nodes_):
                start, end = bounds[node_id], bounds[node_id + 1]
                if end <= start:                       # no training row landed here -> leaf stays 0
                    continue
                idx = row_order[start:end]
                wg = float(np.sum(w[idx] * g[idx]))
                wh = float(np.sum(w[idx] * h[idx]))
                if self.leaf_solver == "newton" and wh > 1e-12:
                    tree.value[node_id] = -wg / wh                 # one Newton step (sklearn's "preset")
                else:
                    ww = float(np.sum(w[idx]))
                    tree.value[node_id] = float(np.sum(w[idx] * r_pseudo[idx]) / ww) if ww > 1e-12 else 0.0
            self.trees.append(tree)

            F = F + self.learning_rate * tree.value[leaf]
            self.train_loss.append(logloss(y, sigmoid(F)))

            # 7. validation curve + early stopping
            if eval_binned:
                Xe, ye = eval_binned[0]
                Fe = np.full(Xe.shape[0], self.base_score_)
                for t in self.trees:
                    Fe = Fe + self.learning_rate * t.value[t.apply(Xe)]
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
        gain = np.sum([t.gain_per_feature_ for t in self.trees], axis=0)
        self.feature_importances_ = gain / gain.sum() if gain.sum() > 0 else gain
        return self

    # -- prediction -----------------------------------------------------------
    def decision_function(self, X, iteration_limit=None):
        """
        Raw score F for each row.  After early stopping the default prediction walks only the trees up to
        `best_iteration_` (same behaviour as xgboost); pass iteration_limit=self.n_estimators_ to use all.
        """
        if iteration_limit is None and getattr(self, "early_stopping_used_", False):
            iteration_limit = self.best_iteration_
        Xb = self._bin(np.asarray(X, dtype=float))
        trees = self.trees if iteration_limit is None else self.trees[:iteration_limit]
        F = np.full(Xb.shape[0], self.base_score_)
        for tree in trees:
            F = F + self.learning_rate * tree.value[tree.apply(Xb)]
        return F

    def predict_proba(self, X, iteration_limit=None):
        """[P(y=0), P(y=1)] -- same layout as scikit-learn."""
        p1 = sigmoid(self.decision_function(X, iteration_limit))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, threshold=0.5):
        """0/1 prediction.  Pass model.best_threshold_ (chosen on validation) for a sensible cut."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def staged_decision_function(self, X):
        """Yield the score after 1, 2, 3, ... trees (used for the performance curve)."""
        Xb = self._bin(np.asarray(X, dtype=float))
        F = np.full(Xb.shape[0], self.base_score_)
        for tree in self.trees:
            F = F + self.learning_rate * tree.value[tree.apply(Xb)]
            yield F.copy()

    def staged_predict_proba(self, X):
        for F in self.staged_decision_function(X):
            yield sigmoid(F)

    # -- helpers used by the report -------------------------------------------
    def feature_importance_table(self, top=None, kind="gain"):
        """List of (feature name, importance) sorted from most to least important (gain = variance reduction)."""
        values = self.feature_importances_
        order = np.argsort(-values)
        if top is not None:
            order = order[:top]
        return [(self.feature_names_[j], float(values[j])) for j in order]

    def pick_threshold(self, X_val, y_val, metric="f1"):
        """
        Choose the decision threshold on the VALIDATION set (never on the test set) using the shared scan
        in Common/metrics_classification.py; the value is stored in `best_threshold_`.
        """
        proba = self.predict_proba(X_val)[:, 1]
        best_thr, best_score = best_threshold(y_val, proba, metric=metric)
        self.best_threshold_ = best_thr
        self.best_threshold_score_ = best_score
        return best_thr
