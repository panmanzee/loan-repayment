
import os
import sys
import time

import numpy as np

_COMMON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Common")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

from metrics_classification import (accuracy, average_precision, best_threshold,logloss, roc_auc)


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


def _impurity(w0, w1, criterion="gini"):
    """Impurity of a node from its weighted class masses.  0 = pure, higher = mixed."""
    w = w0 + w1
    with np.errstate(divide="ignore", invalid="ignore"):
        p0, p1 = np.where(w > 0, w0 / w, 0.0), np.where(w > 0, w1 / w, 0.0)
    if criterion == "gini":
        return 1.0 - p0 * p0 - p1 * p1
    # entropy (natural log, like scikit-learn's xlogx)
    return -(np.where(p0 > 0, p0 * np.log(p0), 0.0) + np.where(p1 > 0, p1 * np.log(p1), 0.0))



class RandomForestTreeScratch:

    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1, min_impurity_decrease=0.0,
                 max_bins=256):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.max_bins = int(max_bins)

    def _new_node(self, proba=0.0):
        self.feature.append(-1)             # -1 = leaf
        self.threshold_bin.append(0)
        self.left.append(-1)
        self.right.append(-1)
        self.leaf_proba.append(float(proba))     
        self.n_rows.append(0)
        self.impurity.append(0.0)
        self.gain.append(0.0)            
        self.node_features_.append(np.empty(0, dtype=int))
        return len(self.leaf_proba) - 1

    def _best_split(self, Xb, w0, w1, rows, W_L0, W_L1, feats):
        
        W = float(self._node_W)                    
        parent = float(_impurity(W_L0.sum(), W_L1.sum(), self.criterion))
        best, best_gain = None, 0.0
        n_rows = len(rows)

        for j in feats:
            b = Xb[rows, j]
            nb = int(b.max()) + 1                         
            h0 = np.bincount(b, weights=w0[rows], minlength=nb)
            h1 = np.bincount(b, weights=w1[rows], minlength=nb)
            n_hist = np.bincount(b, minlength=nb)

            L0 = np.cumsum(h0)[:-1]
            L1 = np.cumsum(h1)[:-1]
            N_L = np.cumsum(n_hist)[:-1]
            R0 = W_L0.sum() - L0
            R1 = W_L1.sum() - L1
            N_R = n_rows - N_L

            valid = (N_L >= self.min_samples_leaf) & (N_R >= self.min_samples_leaf)
            if not valid.any():
                continue

            wl, wr = L0 + L1, R0 + R1
            imp_l = _impurity(L0, L1, self.criterion)
            imp_r = _impurity(R0, R1, self.criterion)
            gain = parent - (wl / max(W, 1e-12)) * imp_l - (wr / max(W, 1e-12)) * imp_r
            gain = np.where(valid, gain, -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > best_gain + 1e-15:
                best_gain = float(gain[k])
                best = (int(j), k, float(gain[k]))
        return best

    def build(self, Xb, y, w, rows, rand, max_features_count):
        
        self.n_features_ = Xb.shape[1]
        self.feature, self.threshold_bin, self.left, self.right = [], [], [], []
        self.leaf_proba, self.n_rows, self.impurity, self.gain = [], [], [], []
        self.node_features_ = []
        self.gain_per_feature_ = np.zeros(self.n_features_)
        self.total_gain_ = 0.0

        w_pos = w * y
        w_neg = w * (1 - y)

        root = self._new_node()
        stack = [(np.asarray(rows, dtype=int), 0, root)]
        while stack:
            node_rows, depth, node = stack.pop()
            self.n_rows[node] = len(node_rows)
            W0 = float(w_neg[node_rows].sum())
            W1 = float(w_pos[node_rows].sum())
            self._node_W = W0 + W1
            self.leaf_proba[node] = W1 / self._node_W if self._node_W > 0 else 0.0
            imp = float(_impurity(W0, W1, self.criterion))
            self.impurity[node] = imp

            if imp <= 0.0:                                     
                continue
            if self.max_depth is not None and depth >= self.max_depth:
                continue
            if len(node_rows) < 2 * self.min_samples_leaf:
                continue

            k_feat = max(1, min(max_features_count, Xb.shape[1]))
            feats = rand.choice(Xb.shape[1], size=k_feat, replace=False)
            self.node_features_[node] = feats

            split = self._best_split(Xb, w_neg, w_pos, node_rows, w_neg[node_rows], w_pos[node_rows], feats)
            if split is None or split[2] <= self.min_impurity_decrease:
                continue

            j, k, gain = split
            self.feature[node] = j
            self.threshold_bin[node] = k
            self.gain[node] = gain
            self.gain_per_feature_[j] += self._node_W * gain         
            self.total_gain_ += self._node_W * gain

            go_left = Xb[node_rows, j] <= k
            left_child = self._new_node()
            right_child = self._new_node()
            self.left[node] = left_child
            self.right[node] = right_child
            stack.append((node_rows[go_left], depth + 1, left_child))
            stack.append((node_rows[~go_left], depth + 1, right_child))

        for name in ("feature", "threshold_bin", "left", "right", "leaf_proba", "n_rows",
                     "impurity", "gain"):
            setattr(self, name, np.asarray(getattr(self, name)))
        self.n_nodes_ = len(self.leaf_proba)
        self.n_leaves_ = int(np.sum(self.feature < 0))
        self.depth_ = self._depth()
        return self

    def _depth(self):
        
        best = 0
        stack = [(0, 0)]
        while stack:
            node, d = stack.pop()
            best = max(best, d)
            if self.feature[node] >= 0:
                stack.append((int(self.left[node]), d + 1))
                stack.append((int(self.right[node]), d + 1))
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

    def predict_proba(self, Xb):
        
        return self.leaf_proba[self.apply(Xb)]


class RandomForestClassifierScratch:

    def __init__(self, n_estimators=100, criterion="gini", max_depth=None, min_samples_leaf=1,
                 max_features="sqrt", bootstrap=True, max_samples=None, class_weight=None,
                 scale_pos_weight="auto", max_bins=256, random_state=42):
        if criterion not in ("gini", "entropy"):
            raise ValueError('criterion must be "gini" or "entropy"')
        if isinstance(scale_pos_weight, str) and scale_pos_weight.lower() != "auto":
            raise ValueError('scale_pos_weight must be "auto" or a number')
        self.n_estimators = int(n_estimators)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.max_samples = max_samples
        self.class_weight = class_weight
        self.scale_pos_weight = scale_pos_weight
        self.max_bins = int(max_bins)
        self.random_state = random_state

    def _make_bin_edges(self, column):
        probs = np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]
        return np.unique(np.quantile(column, probs))

    def _bin(self, X):
        Xb = np.empty(X.shape, dtype=np.int32)
        for j in range(X.shape[1]):
            Xb[:, j] = np.digitize(X[:, j], self.bin_edges_[j])
        return Xb

    def _max_features_count(self, p):
       
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

    def _sample_weights(self, y):
        if isinstance(self.scale_pos_weight, str) and self.scale_pos_weight.lower() == "auto":
            self.scale_pos_weight_ = sqrt_scale_pos_weight(y)
        else:
            self.scale_pos_weight_ = float(self.scale_pos_weight)
        w = np.where(y == 1, self.scale_pos_weight_, 1.0).astype(float)
        if self.class_weight is not None:
            n0, n1 = int((y == 0).sum()), int((y == 1).sum())
            if n1 == 0 or n0 == 0:
                return w
            if self.class_weight == "balanced":
                w = w * np.where(y == 1, len(y) / (2.0 * n1), len(y) / (2.0 * n0))
            elif self.class_weight == "balanced_subsample":
                pass          # applied per bootstrap sample inside fit()
            else:
                raise ValueError('class_weight must be None, "balanced" or "balanced_subsample"')
        return w

  
    def fit(self, X, y, eval_set=None, feature_names=None, early_stopping_rounds=None,
            eval_metric="logloss", verbose=False):
        t0 = time.time()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"f{j}" for j in range(p)]

        rand = np.random.RandomState(self.random_state)
        self.bin_edges_ = [self._make_bin_edges(X[:, j]) for j in range(p)]
        Xb = self._bin(X)
        eval_binned = None
        if eval_set is not None:
            eval_binned = [(self._bin(np.asarray(Xe, dtype=float)), np.asarray(ye, dtype=int))
                           for Xe, ye in eval_set]

        w_base = self._sample_weights(y)
        mf_count = self._max_features_count(p)
        n_boot = n if self.max_samples is None else max(1, int(round(self.max_samples * n)))

        self.trees = []
        self.in_bag_ = np.zeros((self.n_estimators, n), dtype=bool)
        val_sum = np.zeros(eval_binned[0][0].shape[0]) if eval_binned else None
        self.evals_result_ = {"validation": {eval_metric: []}} if eval_binned else {}
        self.best_iteration_ = self.n_estimators
        self.best_score_ = None

        for t in range(self.n_estimators):
            if self.bootstrap:
                idx = rand.randint(0, n, size=n_boot)     
                in_bag = np.zeros(n, dtype=bool)
                in_bag[np.unique(idx)] = True             
            else:
                idx = np.arange(n)
                in_bag = np.ones(n, dtype=bool)

            w_t = w_base
            if self.class_weight == "balanced_subsample":
                yb = y[idx]
                n0, n1 = int((yb == 0).sum()), int((yb == 1).sum())
                if n0 and n1:
                    w_t = w_base * np.where(y == 1, len(yb) / (2.0 * n1), len(yb) / (2.0 * n0))

            tree = RandomForestTreeScratch(criterion=self.criterion, max_depth=self.max_depth,
                                           min_samples_leaf=self.min_samples_leaf,
                                           max_bins=self.max_bins)
            tree.build(Xb, y, w_t, idx, rand, mf_count)
            self.trees.append(tree)
            self.in_bag_[t] = in_bag

        
            if eval_binned:
                Xe, ye = eval_binned[0]
                val_sum = val_sum + tree.predict_proba(Xe)
                proba_va = val_sum / (t + 1)
                score = (logloss(ye, proba_va) if eval_metric == "logloss"
                         else -average_precision(ye, proba_va))
                self.evals_result_["validation"][eval_metric].append(float(score))
                if self.best_score_ is None or score < self.best_score_ - 1e-12:
                    self.best_score_ = score
                    self.best_iteration_ = t + 1
                elif (early_stopping_rounds is not None
                      and (t + 1) - self.best_iteration_ >= early_stopping_rounds):
                    if verbose:
                        print(f"  early stopping at tree {t + 1} (best {self.best_iteration_}, "
                              f"{eval_metric}={self.best_score_:.5f})")
                    break
            if verbose and (t + 1) % 25 == 0:
                print(f"  tree {t + 1:>4} | leaves/tree {np.mean([x.n_leaves_ for x in self.trees]):.0f}")

        self.n_estimators_ = len(self.trees)
        self.fit_time_ = time.time() - t0

    
        P = np.vstack([t.predict_proba(Xb) for t in self.trees]).astype(np.float64)  
        self.proba_matrix_ = P
        self.oob_decision_function_ = self._oob_probabilities(P)
        oob_mask = ~np.isnan(self.oob_decision_function_)
        self.oob_score_ = (float(accuracy(y[oob_mask], self.oob_decision_function_[oob_mask]))
                           if oob_mask.any() else float("nan"))
        self.oob_metrics_ = {}
        if oob_mask.any():
            y_oob, p_oob = y[oob_mask], self.oob_decision_function_[oob_mask]
            self.oob_metrics_ = {"logloss": logloss(y_oob, p_oob), "roc_auc": roc_auc(y_oob, p_oob),
                                 "pr_auc": average_precision(y_oob, p_oob),
                                 "n_oob_rows": int(oob_mask.sum())}

        gain = np.sum([t.gain_per_feature_ for t in self.trees], axis=0)
        self.feature_importances_ = gain / gain.sum() if gain.sum() > 0 else gain
        return self

    def _oob_probabilities(self, P):
    
        seen = self.in_bag_[:len(self.trees)]                 
        counts = (~seen).sum(axis=0)
        sums = (P * (~seen)).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            oob = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        return oob

    # -- prediction -----------------------------------------------------------
    def predict_proba(self, X, iteration_limit=None):

        X = np.asarray(X, dtype=float)
        trees = self.trees if iteration_limit is None else self.trees[:iteration_limit]
        Xb = self._bin(X)
        p1 = np.mean([t.predict_proba(Xb) for t in trees], axis=0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, threshold=0.5):
       
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def staged_predict_proba(self, X):

        Xb = self._bin(np.asarray(X, dtype=float))
        running = np.zeros(Xb.shape[0])
        for k, tree in enumerate(self.trees, 1):
            running = running + tree.predict_proba(Xb)
            p1 = running / k
            yield np.column_stack([1.0 - p1, p1])

    def staged_decision_function(self, X):
        
        for proba in self.staged_predict_proba(X):
            yield proba[:, 1]

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
