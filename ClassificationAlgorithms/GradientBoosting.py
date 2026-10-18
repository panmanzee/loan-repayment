
import os
import sys
import time

import numpy as np


_COMMON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Common")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

from metrics_classification import (accuracy, average_precision, best_threshold,logloss)


def sigmoid(F):
    """Stable 1 / (1 + exp(-F))."""
    F = np.clip(F, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-F))


def sqrt_scale_pos_weight(y):
    """
    The class-imbalance weight :
        raw_weight       = (n_rows - n_positive) / n_positive
        scale_pos_weight = sqrt(raw_weight)
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
            h = np.maximum(proba * (1.0 - proba), 1e-12)      
            g = proba - y                                     
            r_pseudo = -g / h                                

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
            bounds = np.searchsorted(leaf_sorted, np.arange(tree.n_nodes_ + 1))   
            for node_id in range(tree.n_nodes_):
                start, end = bounds[node_id], bounds[node_id + 1]
                if end <= start:                    
                    continue
                idx = row_order[start:end]
                wg = float(np.sum(w[idx] * g[idx]))
                wh = float(np.sum(w[idx] * h[idx]))
                if self.leaf_solver == "newton" and wh > 1e-12:
                    tree.value[node_id] = -wg / wh                 
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

        if iteration_limit is None and getattr(self, "early_stopping_used_", False):
            iteration_limit = self.best_iteration_
        Xb = self._bin(np.asarray(X, dtype=float))
        trees = self.trees if iteration_limit is None else self.trees[:iteration_limit]
        F = np.full(Xb.shape[0], self.base_score_)
        for tree in trees:
            F = F + self.learning_rate * tree.value[tree.apply(Xb)]
        return F

    def predict_proba(self, X, iteration_limit=None):
        p1 = sigmoid(self.decision_function(X, iteration_limit))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def staged_decision_function(self, X):

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
        values = self.feature_importances_
        order = np.argsort(-values)
        if top is not None:
            order = order[:top]
        return [(self.feature_names_[j], float(values[j])) for j in order]

    def pick_threshold(self, X_val, y_val, metric="f1"):
        proba = self.predict_proba(X_val)[:, 1]
        best_thr, best_score = best_threshold(y_val, proba, metric=metric)
        self.best_threshold_ = best_thr
        self.best_threshold_score_ = best_score
        return best_thr
