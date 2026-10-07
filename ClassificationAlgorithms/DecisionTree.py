"""
Decision Tree (CART) from scratch - Classification task (target: not.fully.paid)

Everything is written from scratch with NumPy. Data comes from Common/data_classification.py and every
number is measured with Common/metrics_classification.py.

WHAT A DECISION TREE DOES
-------------------------
The tree keeps asking yes/no questions about ONE feature at a time and splits the rows into two groups:

        fico <= 693.5 ?
          |        |
         yes       no
          |         |
    revol.util <= 52.5 ?   installment <= 250.4 ?
       |      |              |        |
     ...    ...            ...      ...
       |        |            |        |
     leaf    leaf          leaf     leaf      each leaf stores P(y=1) of the rows that landed there

Which question is asked?  The one that makes the two groups as PURE as possible, measured by impurity:

    Gini(w0, w1)    = 1 - (w0/W)^2 - (w1/W)^2                     W = w0 + w1   (weighted counts)
    Entropy(w0, w1) = -(p0 log p0 + p1 log p1)                    p = w / W
    gain = impurity(parent) - (W_L/W)·impurity(left) - (W_R/W)·impurity(right)

The tree grows greedily (the best question now, not the globally best tree), stops when a node is pure
or a size/improvement limit is hit, and a leaf predicts the share of positives that reached it.

TWO SPLIT SEARCH MODES (both implemented here)
----------------------------------------------
    max_bins=None  -> EXACT CART: every feature is sorted at every node and every cut between two
                      different values is evaluated. This is textbook CART and matches scikit-learn.
    max_bins=256   -> BINNED (default): features are pre-binned into 256 quantile buckets once, so the
                      search is a bincount + cumsum. Much faster, and the cut positions are quantiles
                      instead of every single value.

HOW THIS DIFFERS FROM THE OTHER MODELS IN THIS FOLDER
-----------------------------------------------------
                          DecisionTree.py        RandomForest.py        GradientBoosting / XGB
    number of trees       1 (baseline)           many, independent      many, sequential
    extra randomness      none (all features)    bootstrap + sqrt(p)    row/feature subsampling
    learns                the labels             the labels             residuals / gradients
    bias / variance       low bias, HIGH variance low variance          low variance
    interpretable rules   YES (export_text)      no (too many trees)    no
This is why the tree alone usually scores worse -- and why the forest fixes it. Compare them in the report.

Class imbalance: like the other models here, `scale_pos_weight="auto"` (default) weights the positive
rows by sqrt((n - n_pos) / n_pos) -- the ML-project/final.ipynb convention -- and this weight enters the
impurity computation, so the questions themselves are chosen to separate the rare positive class.

Example
-------
    import sys; sys.path.insert(0, "Common")
    from data_classification import build_classification_data, load_dataset, train_val_test_split_scratch
    from metrics_classification import evaluate_on_test

    X, y, names = build_classification_data(load_dataset())
    X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test_split_scratch(X, y)

    tree = DecisionTreeClassifierScratch(max_depth=4, min_samples_leaf=20, criterion="gini")
    tree.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], feature_names=names)
    print(tree.export_text(max_depth=3))                      # <- the rules, readable
    report = evaluate_on_test(y_va, tree.predict_proba(X_va)[:, 1], y_te, tree.predict_proba(X_te)[:, 1])
"""
import os
import sys
import time

import numpy as np

_COMMON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Common")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

from metrics_classification import (average_precision, best_threshold, logloss)


def sqrt_scale_pos_weight(y):
    """
    The class-imbalance weight used throughout this project (same two lines as ML-project/final.ipynb):

        raw_weight       = (n_rows - n_positive) / n_positive
        scale_pos_weight = sqrt(raw_weight)

    Applied as a SAMPLE WEIGHT on the positive rows, so it flows into the impurity formula above.
    """
    y = np.asarray(y, dtype=int).ravel()
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return 1.0
    return float(np.sqrt((len(y) - n_pos) / n_pos))


def _impurity(w0, w1, criterion="gini"):
    """Impurity of a node from its weighted class masses. 0 = pure, higher = more mixed."""
    w = w0 + w1
    with np.errstate(divide="ignore", invalid="ignore"):
        p0 = np.where(w > 0, w0 / w, 0.0)
        p1 = np.where(w > 0, w1 / w, 0.0)
    if criterion == "gini":
        return 1.0 - p0 * p0 - p1 * p1
    return -(np.where(p0 > 0, p0 * np.log(p0), 0.0) + np.where(p1 > 0, p1 * np.log(p1), 0.0))


# -----------------------------------------------------------------------------
# The tree
# -----------------------------------------------------------------------------
class DecisionTreeScratch:
    """
    Binary CART classification tree.

    Parameters
    ----------
    criterion              "gini" (default) or "entropy"
    max_depth              maximum depth (None = grow until pure / blocked by the size limits)
    min_samples_split      a node needs at least this many ROWS to be considered for a split
    min_samples_leaf       every child must keep at least this many ROWS
    max_features           features considered at each split: None (default, all), "sqrt", "log2",
                           an int, or a float fraction. nit: a single tree normally uses ALL features --
                           restricting them is what RandomForest.py does per node.
    min_impurity_decrease  a split must reduce the impurity at least by this (0 = only gain > 0)
    max_bins               None = exact CART split search on the raw values, else quantile buckets

    Fitted attributes
    -----------------
    feature, threshold, left, right   the structure (-1 in `feature` means leaf)
    leaf_proba                        P(y=1) of each node (a leaf's prediction)
    n_rows, impurity, gain            diagnostics per node
    depth_, n_leaves_, n_nodes_, feature_importances_
    """

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, min_impurity_decrease=0.0, max_bins=256, random_state=None):
        if criterion not in ("gini", "entropy"):
            raise ValueError('criterion must be "gini" or "entropy"')
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.max_bins = max_bins
        self.random_state = random_state

    # -- node bookkeeping -----------------------------------------------------
    def _new_node(self, proba=0.0):
        self.feature.append(-1)              # -1 = leaf
        self.threshold.append(0.0)
        self.left.append(-1)
        self.right.append(-1)
        self.leaf_proba.append(float(proba))
        self.n_rows.append(0)
        self.impurity.append(0.0)
        self.gain.append(0.0)
        return len(self.leaf_proba) - 1

    # -- feature handling -----------------------------------------------------
    def _feature_count(self, p):
        mf = self.max_features
        if mf is None:
            return p
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(np.sqrt(p)))
            if mf == "log2":
                return max(1, int(np.log2(p)))
            raise ValueError('max_features string must be "sqrt" or "log2"')
        if isinstance(mf, float):
            return max(1, int(round(mf * p)))
        return max(1, min(int(mf), p))

    def _make_bin_edges(self, column):
        probs = np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]
        return np.unique(np.quantile(column, probs))

    # -- split search: binned -------------------------------------------------
    def _best_split_binned(self, Xb, w0, w1, rows, feats, total_w, parent):
        best, best_gain = None, 0.0
        n_rows = len(rows)
        for j in feats:
            b = Xb[rows, j]
            nb = int(b.max()) + 1                       # only the bins present in this node
            h0 = np.bincount(b, weights=w0[rows], minlength=nb)
            h1 = np.bincount(b, weights=w1[rows], minlength=nb)
            n_hist = np.bincount(b, minlength=nb)

            L0, L1 = np.cumsum(h0)[:-1], np.cumsum(h1)[:-1]
            N_L = np.cumsum(n_hist)[:-1]
            R0, R1 = w0[rows].sum() - L0, w1[rows].sum() - L1
            N_R = n_rows - N_L

            valid = (N_L >= self.min_samples_leaf) & (N_R >= self.min_samples_leaf)
            if not valid.any():
                continue
            wl, wr = L0 + L1, R0 + R1
            gain = parent - (wl / total_w) * _impurity(L0, L1, self.criterion) \
                          - (wr / total_w) * _impurity(R0, R1, self.criterion)
            gain = np.where(valid, gain, -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > best_gain + 1e-15:
                best_gain, best = float(gain[k]), (int(j), float(k), float(gain[k]))
        return best

    # -- split search: exact (textbook CART) ----------------------------------
    def _best_split_exact(self, X, w0, w1, rows, feats, total_w, parent):
        best, best_gain = None, 0.0
        n_rows = len(rows)
        u0, u1 = w0[rows], w1[rows]
        for j in feats:
            x = X[rows, j]
            order = np.argsort(x, kind="stable")
            xs = x[order]
            a0, a1 = u0[order], u1[order]
            L0, L1 = np.cumsum(a0)[:-1], np.cumsum(a1)[:-1]
            N_L = np.arange(1, n_rows)
            N_R = n_rows - N_L
            R0, R1 = L0[-1] + a0[-1] - L0, L1[-1] + a1[-1] - L1
            distinct = xs[1:] != xs[:-1]                # a cut is only possible between two values
            valid = distinct & (N_L >= self.min_samples_leaf) & (N_R >= self.min_samples_leaf)
            if not valid.any():
                continue
            wl, wr = L0 + L1, R0 + R1
            gain = parent - (wl / total_w) * _impurity(L0, L1, self.criterion) \
                          - (wr / total_w) * _impurity(R0, R1, self.criterion)
            gain = np.where(valid, gain, -np.inf)
            i = int(np.argmax(gain))
            if gain[i] > best_gain + 1e-15:
                best_gain = float(gain[i])
                best = (int(j), float((xs[i] + xs[i + 1]) / 2.0), float(gain[i]))   # midpoint cut
        return best

    # -- fit the tree ---------------------------------------------------------
    def fit(self, X, y, sample_weight=None, feature_names=None, verbose=False):
        """
        Grow the tree (this is the tree's own fit; DecisionTreeClassifierScratch wraps it for the project
        interface). `sample_weight` lets the caller pass the class weights in.
        """
        t0 = time.time()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"f{j}" for j in range(p)]
        self.binned_ = self.max_bins is not None
        rand = np.random.RandomState(self.random_state)
        self.rand_ = rand

        if sample_weight is None:
            w = np.ones(n)
        else:
            w = np.asarray(sample_weight, dtype=float).ravel()

        if self.binned_:
            self.bin_edges_ = [self._make_bin_edges(X[:, j]) for j in range(p)]
            Xb = np.empty(X.shape, dtype=np.int32)
            for j in range(p):
                Xb[:, j] = np.digitize(X[:, j], self.bin_edges_[j])
        else:
            self.bin_edges_ = None
            Xb = None

        w_pos = w * y
        w_neg = w * (1 - y)
        n_feats = self._feature_count(p)

        self.feature, self.threshold, self.left, self.right = [], [], [], []
        self.leaf_proba, self.n_rows, self.impurity, self.gain = [], [], [], []
        self.gain_per_feature_ = np.zeros(p)
        self.total_gain_ = 0.0
        self.depth_ = 0

        root = self._new_node()
        stack = [(np.arange(n, dtype=int), 0, root)]
        while stack:
            node_rows, depth, node = stack.pop()
            W0 = float(w_neg[node_rows].sum())
            W1 = float(w_pos[node_rows].sum())
            W = W0 + W1
            self.n_rows[node] = len(node_rows)
            self.leaf_proba[node] = W1 / W if W > 0 else 0.0
            imp = float(_impurity(W0, W1, self.criterion))
            self.impurity[node] = imp
            self.depth_ = max(self.depth_, depth)

            if imp <= 0.0:                                            # pure -> leaf
                continue
            if self.max_depth is not None and depth >= self.max_depth:
                continue
            if len(node_rows) < self.min_samples_split or len(node_rows) < 2 * self.min_samples_leaf:
                continue

            feats = (rand.choice(p, size=n_feats, replace=False) if n_feats < p else np.arange(p))
            if self.binned_:
                split = self._best_split_binned(Xb, w_neg, w_pos, node_rows, feats, W, imp)
            else:
                split = self._best_split_exact(X, w_neg, w_pos, node_rows, feats, W, imp)
            if split is None or split[2] <= self.min_impurity_decrease:
                continue

            j, thr, gain = split
            self.feature[node] = j
            self.threshold[node] = thr
            self.gain[node] = gain
            self.gain_per_feature_[j] += W * gain
            self.total_gain_ += W * gain

            if self.binned_:
                go_left = Xb[node_rows, j] <= thr
            else:
                go_left = X[node_rows, j] <= thr
            left_child = self._new_node()
            right_child = self._new_node()
            self.left[node] = left_child
            self.right[node] = right_child
            stack.append((node_rows[go_left], depth + 1, left_child))
            stack.append((node_rows[~go_left], depth + 1, right_child))

        for name in ("feature", "threshold", "left", "right", "leaf_proba", "n_rows", "impurity",
                     "gain"):
            setattr(self, name, np.asarray(getattr(self, name)))
        self.n_nodes_ = len(self.leaf_proba)
        self.n_leaves_ = int(np.sum(self.feature < 0))
        self.fit_time_ = time.time() - t0
        gain = self.gain_per_feature_
        self.feature_importances_ = gain / gain.sum() if gain.sum() > 0 else gain
        if verbose:
            print(f"  tree grown: depth {self.depth_}, leaves {self.n_leaves_}, "
                  f"nodes {self.n_nodes_}, {self.fit_time_:.2f}s")
        return self

    # -- prediction -----------------------------------------------------------
    def _transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.binned_:
            Xb = np.empty(X.shape, dtype=np.int32)
            for j in range(X.shape[1]):
                Xb[:, j] = np.digitize(X[:, j], self.bin_edges_[j])
            return Xb
        return X

    def apply(self, X, depth_limit=None):
        """
        Node index every row ends up in (the leaf, or the node where a depth limit stopped the walk).
        Handy for inspecting the tree: np.bincount(model.tree_.apply(X_train)).
        """
        Xv = self._transform(X)
        node = np.zeros(Xv.shape[0], dtype=int)
        depth = np.zeros(Xv.shape[0], dtype=int)
        while True:
            feat = self.feature[node]
            internal = feat >= 0
            if depth_limit is not None:
                internal = internal & (depth < depth_limit)
            if not internal.any():
                break
            r = np.where(internal)[0]
            go_left = Xv[r, feat[r]] <= self.threshold[node[r]]
            node[r] = np.where(go_left, self.left[node[r]], self.right[node[r]])
            depth[r] += 1
        return node

    def predict_proba(self, X, depth_limit=None):
        """
        [P(y=0), P(y=1)] for every row -- the positive share of the leaf it reaches.
        `depth_limit=k` answers with a shallower tree (used for the depth/overfitting curve).
        """
        p1 = self.leaf_proba[self.apply(X, depth_limit=depth_limit)]
        return np.column_stack([1.0 - p1, p1])

    def export_text(self, max_depth=None, indent="    "):
        """
        Render the tree as readable if/else rules. Because a single tree IS interpretable, this is the
        part of a decision-tree report that a forest cannot give you.
        """
        lines = []

        def walk(node, depth, prefix):
            name = self.feature_names_[self.feature[node]] if self.feature[node] >= 0 else None
            if name is None or (max_depth is not None and depth >= max_depth):
                lines.append(f"{prefix}-> P(y=1) = {self.leaf_proba[node]:.4f} "
                             f"(n={self.n_rows[node]})")
                return
            thr = self.threshold[node]
            if self.binned_:
                # the split is stored as a bin index -> show the raw cut value (a bin edge) instead
                edges = self.bin_edges_[int(self.feature[node])]
                thr = float(edges[int(thr)]) if int(thr) < len(edges) else float("inf")
            lines.append(f"{prefix}if {name} <= {thr:.4f}:")
            walk(int(self.left[node]), depth + 1, prefix + indent)
            lines.append(f"{prefix}else:")
            walk(int(self.right[node]), depth + 1, prefix + indent)

        walk(0, 0, "")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Project-facing wrapper (same interface as RandomForest.py / GradientBoosting.py / XGB.py)
# -----------------------------------------------------------------------------
class DecisionTreeClassifierScratch:
    """
    Decision tree classifier with the project interface.

    Parameters
    ----------
    criterion              "gini" (default) or "entropy"
    max_depth              None = grow until pure (a single tree then overfits; that is the lesson)
    min_samples_split      minimum ROWS to consider splitting a node (default 2)
    min_samples_leaf       minimum ROWS in a child (raise this to regularise the tree)
    max_features           None (default, all features), "sqrt", "log2", int or float fraction
    min_impurity_decrease  minimum impurity reduction required for a split
    max_bins               None = exact CART split search, else quantile buckets (default 256)
    class_weight           None (default) or "balanced" (scikit-learn style weights)
    scale_pos_weight       project convention: "auto" (default) = sqrt((n - n_pos) / n_pos) from y_train,
                           or a number, or 1.0 to switch the extra positive weighting off
    random_state           seed (only used when max_features restricts the feature set)

    Fitted attributes
    -----------------
    tree_ (the DecisionTreeScratch), feature_names_, feature_importances_, scale_pos_weight_,
    depth_, n_leaves_, best_depth_, evals_result_, fit_time_
    """

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, min_impurity_decrease=0.0, max_bins=256, class_weight=None,
                 scale_pos_weight="auto", random_state=42):
        if isinstance(scale_pos_weight, str) and scale_pos_weight.lower() != "auto":
            raise ValueError('scale_pos_weight must be "auto" or a number')
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.max_bins = max_bins
        self.class_weight = class_weight
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state

    def _sample_weights(self, y):
        if isinstance(self.scale_pos_weight, str) and self.scale_pos_weight.lower() == "auto":
            self.scale_pos_weight_ = sqrt_scale_pos_weight(y)
        else:
            self.scale_pos_weight_ = float(self.scale_pos_weight)
        w = np.where(y == 1, self.scale_pos_weight_, 1.0).astype(float)
        if self.class_weight is not None:
            n0, n1 = int((y == 0).sum()), int((y == 1).sum())
            if self.class_weight == "balanced" and n0 and n1:
                w = w * np.where(y == 1, len(y) / (2.0 * n1), len(y) / (2.0 * n0))
            elif self.class_weight != "balanced":
                raise ValueError('class_weight must be None or "balanced"')
        return w

    def fit(self, X, y, eval_set=None, feature_names=None, early_stopping_rounds=None,
            eval_metric="logloss", verbose=False):
        """
        Grow the tree.

        When `eval_set` is given, the validation metric is measured at EVERY DEPTH and stored in
        `evals_result_` together with `best_depth_`. That curve is the classic overfitting picture: a
        single tree keeps improving on the training rows while the validation score turns around and gets
        worse. If `early_stopping_rounds` is also passed, predictions default to `best_depth_`.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"f{j}" for j in range(X.shape[1])]
        w = self._sample_weights(y)

        self.tree_ = DecisionTreeScratch(
            criterion=self.criterion, max_depth=self.max_depth,
            min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features, min_impurity_decrease=self.min_impurity_decrease,
            max_bins=self.max_bins, random_state=self.random_state)
        self.tree_.fit(X, y, sample_weight=w, feature_names=self.feature_names_, verbose=verbose)
        self.feature_importances_ = self.tree_.feature_importances_
        self.depth_ = self.tree_.depth_
        self.n_leaves_ = self.tree_.n_leaves_
        self.fit_time_ = self.tree_.fit_time_

        # train/validation curve per depth (single tree -> depth is the "iteration" axis)
        self.train_loss_ = []
        for d in range(1, self.depth_ + 1):
            self.train_loss_.append(logloss(y, self.tree_.predict_proba(X, depth_limit=d)[:, 1]))

        self.evals_result_ = {}
        self.best_depth_ = self.depth_
        self.early_stopping_used_ = False
        if eval_set is not None:
            X_va, y_va = eval_set[0]
            X_va = np.asarray(X_va, dtype=float)
            y_va = np.asarray(y_va, dtype=int)
            curve = []
            for d in range(1, self.depth_ + 1):
                p = self.tree_.predict_proba(X_va, depth_limit=d)[:, 1]
                curve.append(float(logloss(y_va, p) if eval_metric == "logloss"
                                   else -average_precision(y_va, p)))
            self.evals_result_ = {"validation": {eval_metric: curve}}
            if curve:
                self.best_depth_ = int(np.argmin(curve)) + 1
                self.best_score_ = float(np.min(curve))
            if early_stopping_rounds is not None:
                self.early_stopping_used_ = True
            if verbose:
                print(f"  depth curve: best depth {self.best_depth_} "
                      f"({eval_metric}={self.best_score_:.5f}), tree depth {self.depth_}")
        return self

    # -- prediction -----------------------------------------------------------
    def _depth_limit(self):
        """After early stopping, answer with the depth that was best on validation (like the other models)."""
        return self.best_depth_ if getattr(self, "early_stopping_used_", False) else None

    def predict_proba(self, X, depth_limit=None):
        limit = self._depth_limit() if depth_limit is None else depth_limit
        return self.tree_.predict_proba(X, depth_limit=limit)

    def predict(self, X, threshold=0.5):
        """0/1 prediction.  Pass model.best_threshold_ (chosen on validation) for a sensible cut."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def staged_predict_proba(self, X):
        """
        Yield [P(0), P(1)] for depth 1, 2, 3, ... -- the tree answered by a shallower and shallower cut
        of itself. Plotting this against the validation loss is the standard overfitting figure.
        """
        for d in range(1, self.depth_ + 1):
            yield self.tree_.predict_proba(X, depth_limit=d)

    def staged_decision_function(self, X):
        for proba in self.staged_predict_proba(X):
            yield proba[:, 1]

    # -- helpers used by the report -------------------------------------------
    def apply(self, X, depth_limit=None):
        """Node index of every row (see DecisionTreeScratch.apply)."""
        return self.tree_.apply(X, depth_limit=depth_limit)

    def export_text(self, max_depth=None):
        """Readable if/else rules of the fitted tree (the interpretability artefact)."""
        return self.tree_.export_text(max_depth=max_depth)

    def feature_importance_table(self, top=None, kind="gain"):
        """List of (feature name, importance) sorted from most to least important (weighted impurity decrease)."""
        values = self.feature_importances_
        order = np.argsort(-values)
        if top is not None:
            order = order[:top]
        return [(self.feature_names_[j], float(values[j])) for j in order]

    def pick_threshold(self, X_val, y_val, metric="f1"):
        """
        Choose the decision threshold on the VALIDATION set (never on the test set) with the shared scan
        in Common/metrics_classification.py; stored in `best_threshold_`.
        """
        proba = self.predict_proba(X_val)[:, 1]
        best_thr, best_score = best_threshold(y_val, proba, metric=metric)
        self.best_threshold_ = best_thr
        self.best_threshold_score_ = best_score
        return best_thr
