"""
k-Nearest Neighbors (from scratch) - Classification task (target: not.fully.paid, 1 = defaulted)

NumPy only and self-contained (no import from Common/), like the other part-2 files in this folder
(NaiveBayes.py, Perceptron.py, MLP.py, SVC.py, Dimensionreduction.py).

A NOTE ON THE NAME
------------------
k-NN is often described as "clustering by neighbourhood", but it is a SUPERVISED method: unlike
k-means it does use y. The neighbour idea is shared, the labels are not - here every training row is
stored with its label and new rows are voted on by their closest neighbours.

THE IDEA
--------
There is no training at all (a "lazy" learner): fit() just stores the data. To predict a row, look at
the k closest stored rows and let them vote:

                        k = 5 neighbours of the star
              class 0  o o          x x   class 1
                        o       *        x        * = new row
                     o    o          x  x             4 of its 5 neighbours are 'x'
                      o                x x             -> predict class 1, proba 4/5

    distance        euclidean (default), manhattan (p=1) or cosine - all computed from scratch
    weights         "distance" (1/d, a close neighbour counts more) or "uniform" (plain vote)
    k               THE hyper-parameter: too small = follows noise, too big = blurs the boundary.
                    Here k is chosen on the VALIDATION set only (never on the test set), by scanning a
                    grid of odd values and keeping the best ROC-AUC - see `k_selection_`.

WHY SCALING (AND CLIPPING) MATTERS MORE HERE THAN ANYWHERE ELSE
---------------------------------------------------------------
k-NN has no weights: it compares raw distances, so a column measured in thousands dominates one
measured in tenths. Two consequences on this data set, both measured:
    1) standardise the columns (scale=True), otherwise revol.bal (0..1e5) decides every neighbour;
    2) even then the rare one-hot / ratio columns reach z-scores of ~40, and such a row looks "far" from
       everything. clip=5.0 (winsorising) is therefore kept as the default, the same guard the SVM
       needed. Be honest about the size of the effect on THIS data set though: validation ROC goes
       0.6411 (no clip) / 0.6431 (clip 3) / 0.6407 (clip 5) - a tie inside the noise, while scaling is
       worth about 0.01. Keep the clip for safety, not because it is a win here.

PROBABILITY OUTPUT
------------------
The raw score is the (class-weighted) share of the k neighbours that belong to class 1. That number
is already a probability, and it is turned into a calibrated one with Platt scaling fitted on the
validation scores, exactly like the other models in this folder.

Example
-------
    import sys; sys.path.insert(0, "Common")
    from data_classification import build_classification_data, load_dataset, train_val_test_split_scratch
    from metrics_classification import evaluate_on_test
    from KNN import KNNClassifierScratch

    X, y, names = build_classification_data(load_dataset())
    X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test_split_scratch(X, y)

    knn = KNNClassifierScratch(n_neighbors=25, weights="distance")
    knn.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], feature_names=names)   # tunes k on validation
    print(knn.k_selection_["best_k"], knn.k_selection_["val_auc"])

    evaluate_on_test(y_te, knn.predict_proba(X_te)[:, 1], y_val=y_va,
                     proba_val=knn.predict_proba(X_va)[:, 1], selected_threshold=knn.best_threshold_)
"""

import time

import numpy as np


# --------------------------------------------------------------------------------------- helpers
def _sigmoid(z):
    """Stable 1 / (1 + exp(-z)) written with tanh (no overflow warnings)."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))


def sqrt_scale_pos_weight(y):
    """Project-wide imbalance weight for the positive rows: sqrt((n - n_pos) / n_pos)  (~2.29 here)."""
    y = np.asarray(y, dtype=int).ravel()
    n_pos = int(y.sum())
    return 1.0 if n_pos in (0, len(y)) else float(np.sqrt((len(y) - n_pos) / n_pos))


def _weights(y, scale_pos_weight, sample_weight):
    """
    Per-row weights following the folder convention, plus the "balanced" option:

        "auto"     sqrt(n_neg / n_pos)   -> 2.29 on this data (the convention of the other models)
        "balanced" n_neg / n_pos         -> 5.21, the two classes cost the same (default here)
        number     that number

    The weight multiplies the row's vote. For k-NN a constant per-class weight only rescales the score
    and therefore changes no metric once the threshold is chosen on validation (derivation and numbers
    in the class docstring); it is kept for interface compatibility with the other models.
    """
    y = np.asarray(y, dtype=int).ravel()
    n_pos = int(np.sum(y == 1))
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        spw = 1.0
    elif scale_pos_weight == "auto":
        spw = sqrt_scale_pos_weight(y)
    elif scale_pos_weight == "balanced":
        spw = float(n_neg) / n_pos
    else:
        spw = float(scale_pos_weight)
    w = np.where(y == 1, spw, 1.0)
    if sample_weight is not None:
        w = w * np.asarray(sample_weight, dtype=float)
    return w, spw


def _wlogloss(y, p, w):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=w))


def _auc(y, s):
    """ROC-AUC by the rank-sum (Mann-Whitney U) identity, ties get the average rank."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # average the ranks of tied scores
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _average_precision(y, s):
    """PR-AUC (average precision), the metric that matters for a 16%-positive target."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    yy = y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def _default_names(n, prefix="feat"):
    return [f"{prefix}{i + 1}" for i in range(n)]


class _Base:
    """Shared plumbing (same shape as the other files in this folder): scaling, threshold, state."""

    def _fit_scaler(self, X):
        self.mean_ = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        self.scale_ = sd

    def _z(self, X):
        Z = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        # winsorise: cap the z-scores at +-clip. The bound is defined on Z-SCORES, so it is only applied
        # when the columns were standardised (on raw columns it would erase every large-unit feature).
        if self.scale and self.clip is not None:
            Z = np.clip(Z, -float(self.clip), float(self.clip))
        return Z

    def predict(self, X, threshold=None):
        t = self.best_threshold_ if threshold is None else threshold
        return (self.predict_proba(X)[:, 1] >= t).astype(int)

    def pick_threshold(self, X_val, y_val, metric="f1"):
        """Choose the decision threshold on the validation rows (never on the test rows)."""
        p = self.predict_proba(X_val)[:, 1]
        y = np.asarray(y_val).astype(int)
        best_t, best_s = 0.5, -1.0
        for t in np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97))):
            pred = (p >= t).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            s = (2 * tp / max(2 * tp + fp + fn, 1) if metric == "f1"
                 else 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)))
            if s > best_s:
                best_t, best_s = float(t), float(s)
        self.best_threshold_, self.best_threshold_score_ = best_t, best_s
        return best_t

    @property
    def train_loss(self):
        h = getattr(self, "history_", None) or {}
        return h.get("train_logloss")

    @property
    def evals_result_(self):
        h = getattr(self, "history_", None) or {}
        return {"validation": {"logloss": h.get("val_logloss", [])}}


# =============================================================================================== KNN
class KNNClassifierScratch(_Base):
    """
    k-Nearest Neighbors classifier.

    Parameters
    ----------
    n_neighbors       k; the value used when no tuning happens (ignored when k is tuned)
    weights           "distance" (1/d, default) or "uniform"
    metric            "euclidean" (default), "manhattan" or "cosine"
    scale             standardise the columns first (True; k-NN compares distances, so this is not
                      optional in practice)
    clip              winsorise the z-scores at +-clip (default 5.0; None = off, and it only applies
                      together with scale=True because the bound is in standard deviations). See the
                      docstring: without it a single row with a z-score of 40 poisons a neighbourhood
    scale_pos_weight  "balanced" (default) = n_neg/n_pos, "auto" = sqrt(...) as in the other models, a
                      number, or 1.0 for a plain majority vote.
                      Careful, because k-NN is special here: a constant per-class weight CANNOT change
                      what the model predicts, whichever vote you use.
                        distance vote: p = cA / (cA + B) with A = sum of 1/d over the positive
                                       neighbours, B = the same over the negative ones - so the ranking
                                       of p is the ranking of A/B, with c cancelled out;
                        uniform vote:  p = c n_pos / (c n_pos + n_neg) with n_neg = k - n_pos, which is
                                       a monotone function of the neighbour count n_pos.
                      Both are only a rescaling of the score, and a validation-picked threshold or Platt
                      scaling absorbs a rescaling exactly. Measured on this data: ROC/PR-AUC/F1/logloss
                      are identical to 4 decimals for 1.0 / 2.29 / 5.21. So: the real knobs of k-NN are
                      k and the distance metric (and which columns you feed it), not the class weight.
                      The parameter is kept because the runner passes it like everywhere else.
    tune_k            True/False, or "auto" (default) = tune whenever an eval_set is given
    k_grid            candidate k values (default: odd numbers 1..151, capped by the training size)
    tune_metric       "auc" (default), "ap" or "logloss" - what the k scan maximises/minimises
    block             how many query rows to compare against the training set at a time (memory)

    Fitted attributes
    -----------------
    X_train_, y_train_, cw_, mean_, scale_, n_neighbors_ (the k actually used), k_selection_
    (best_k, best_value, grid, val_auc, val_ap, val_logloss), history_, platt_, best_threshold_,
    feature_names_, feature_importances_, fit_time_, predict_time_
    """

    def __init__(self, n_neighbors=25, weights="distance", metric="euclidean", scale=True, clip=5.0,
                 scale_pos_weight="balanced", tune_k="auto", k_grid=None, tune_metric="auc",
                 block=512, random_state=42, feature_names=None):
        if weights not in ("uniform", "distance"):
            raise ValueError('weights must be "uniform" or "distance"')
        if metric not in ("euclidean", "manhattan", "cosine"):
            raise ValueError('metric must be "euclidean", "manhattan" or "cosine"')
        if isinstance(scale_pos_weight, str) and scale_pos_weight not in ("auto", "balanced"):
            raise ValueError('scale_pos_weight must be "auto", "balanced" or a number')
        if tune_k not in (True, False, "auto"):
            raise ValueError('tune_k must be True, False or "auto"')
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.metric = metric
        self.scale = bool(scale)
        self.clip = clip
        self.scale_pos_weight = scale_pos_weight
        self.tune_k = tune_k
        self.k_grid = k_grid
        self.tune_metric = tune_metric
        self.block = int(block)
        self.random_state = random_state
        self.feature_names = feature_names

    # -------------------------------------------------------------- distances
    def _distances(self, A, B):
        """Full distance matrix between the query rows A (m, d) and the stored rows B (n, d)."""
        if self.metric == "euclidean":
            # ||a-b||^2 = |a|^2 - 2 a.b + |b|^2, the expansion avoids an m x n x d tensor
            aa = np.sum(A * A, axis=1).reshape(-1, 1)
            bb = np.sum(B * B, axis=1).reshape(1, -1)
            return np.sqrt(np.maximum(aa - 2.0 * (A @ B.T) + bb, 0.0))
        if self.metric == "manhattan":
            return np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)
        # cosine: 1 - cos(angle); zero vectors (possible after clipping) get distance 1
        na = np.linalg.norm(A, axis=1, keepdims=True)
        nb = np.linalg.norm(B, axis=1, keepdims=True).T
        denom = np.maximum(na * nb, 1e-12)
        return 1.0 - (A @ B.T) / denom

    def kneighbors(self, X, n_neighbors=None):
        """Positions and distances of the nearest training rows (the sklearn-style accessor)."""
        k = int(n_neighbors or self.n_neighbors_)
        k = max(1, min(k, len(self.X_train_)))
        Z = self._z(X) if self.scale else np.asarray(X, dtype=float)
        out_d = np.empty((len(Z), k))
        out_i = np.empty((len(Z), k), dtype=int)
        for start in range(0, len(Z), self.block):
            chunk = Z[start:start + self.block]
            D = self._distances(chunk, self.X_train_)
            idx = np.argpartition(D, k - 1, axis=1)[:, :k]
            rows = np.arange(len(chunk))[:, None]
            order = np.argsort(D[rows, idx], axis=1)
            idx = idx[rows, order]
            out_i[start:start + len(chunk)] = idx
            out_d[start:start + len(chunk)] = D[rows, idx]
        return out_d, out_i

    def _vote(self, X, k=None):
        """Weighted share of the k neighbours that are positive (the raw score)."""
        k = int(k or self.n_neighbors_)
        D, I = self._raw_kneighbors(self._z(X), k)
        p, _ = self._vote_from_neighbors(D, I)
        return p, self.y_train_[I], I

    # ------------------------------------------------------------------- fit
    def fit(self, X, y, eval_set=None, feature_names=None, early_stopping_rounds=None,
            sample_weight=None, verbose=False, **_ignored):
        """
        Store the training rows (there is nothing to train) and, when an eval_set is given, choose k on
        the validation rows. `early_stopping_rounds` is accepted for interface compatibility and ignored:
        a lazy learner has no iterations to stop.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or inf - handle missing values first")
        y = np.asarray(y).astype(int).ravel()
        if len(y) != len(X):
            raise ValueError("X and y have different lengths")
        t0 = time.time()

        self.n_features_in_ = X.shape[1]
        self.feature_names_ = list(feature_names or self.feature_names or
                                   _default_names(self.n_features_in_))
        if len(self.feature_names_) != self.n_features_in_:
            raise ValueError("feature_names has the wrong length")

        if self.scale:
            self._fit_scaler(X)
        else:
            self.mean_ = np.zeros(X.shape[1])
            self.scale_ = np.ones(X.shape[1])
        Z = self._z(X)
        self.X_train_ = Z.copy()
        self.y_train_ = y.copy()
        self.cw_, self.scale_pos_weight_ = _weights(y, self.scale_pos_weight, sample_weight)
        n_train = len(Z)

        # ---- choose k on the validation set (a LAZY learner has exactly one hyper-parameter)
        Zv = yv = None
        if eval_set is not None:
            Zv = self._z(np.asarray(eval_set[0][0], dtype=float))
            yv = np.asarray(eval_set[0][1]).astype(int).ravel()
        grid = self.k_grid or [1, 3, 5, 7, 9, 11, 15, 21, 25, 31, 41, 51, 71, 101, 151]
        grid = [int(k) for k in grid if 1 <= int(k) <= n_train]
        if not grid:
            grid = [max(1, min(self.n_neighbors, n_train))]
        do_tune = (self.tune_k is True) or (self.tune_k == "auto" and Zv is not None)
        kmax = min(max(grid), n_train)

        hist = {"k": [], "train_logloss": [], "val_logloss": [], "val_auc": [], "val_ap": []}
        if do_tune and Zv is not None:
            D, I = self._raw_kneighbors(Zv, kmax)          # computed once, reused for every k
            best_k, best_val = grid[0], -np.inf
            for k in grid:
                p, _ = self._vote_from_neighbors(D[:, :k], I[:, :k])
                auc = _auc(yv, p)
                ap = _average_precision(yv, p)
                ll = _wlogloss(yv, np.clip(p, 1e-6, 1 - 1e-6), np.ones(len(yv)))
                hist["k"].append(k)
                hist["val_auc"].append(auc)
                hist["val_ap"].append(ap)
                hist["val_logloss"].append(ll)
                hist["train_logloss"].append(self._train_logloss(k))
                val = {"auc": auc, "ap": ap, "logloss": -ll}[self.tune_metric]
                if val > best_val:
                    best_k, best_val = k, val
                if verbose:
                    print(f"  k={k:>4} | val ROC {auc:.4f} | val AP {ap:.4f} | val logloss {ll:.4f}")
            self.n_neighbors_ = int(best_k)
            self.k_selection_ = {"best_k": int(best_k), "best_value": float(best_val),
                                 "metric": self.tune_metric, "grid": grid,
                                 "val_auc": hist["val_auc"], "val_ap": hist["val_ap"],
                                 "val_logloss": hist["val_logloss"]}
        else:
            self.n_neighbors_ = max(1, min(self.n_neighbors, n_train))
            pv = self._vote(Zv if Zv is not None else Z[:min(500, n_train)])[0]
            yv_use = yv if yv is not None else y[:min(500, n_train)]
            hist["k"].append(self.n_neighbors_)
            hist["val_auc"].append(_auc(yv_use, pv))
            hist["val_ap"].append(_average_precision(yv_use, pv))
            hist["val_logloss"].append(_wlogloss(yv_use, np.clip(pv, 1e-6, 1 - 1e-6), np.ones(len(yv_use))))
            hist["train_logloss"].append(self._train_logloss(self.n_neighbors_))
            self.k_selection_ = {"best_k": self.n_neighbors_, "best_value": None, "metric": None,
                                 "grid": [self.n_neighbors_], "val_auc": hist["val_auc"],
                                 "val_ap": hist["val_ap"], "val_logloss": hist["val_logloss"]}
        self.history_ = hist

        # ---- Platt scaling on the validation scores (a better calibration than the training scores)
        if Zv is not None:
            self._fit_platt(self._vote(Zv)[0], yv, np.ones(len(yv)))
            self._platt_eval = (Zv.copy(), yv.copy())
        else:
            S = self._vote(Z)[0]
            self._fit_platt(S, y, self.cw_)
            self._platt_eval = None
        self.best_threshold_ = 0.5
        self.best_threshold_score_ = None
        self.feature_importances_ = None                    # computed lazily (see feature_importance_table)
        self.fitted_ = True
        self.fit_time_ = time.time() - t0
        self.predict_time_ = None
        return self

    def _train_sample(self, n=400):
        """A small slice of the training rows: used for the cheap training-side curves."""
        if len(self.X_train_) <= n:
            return self.X_train_
        return self.X_train_[:n]

    def _train_logloss(self, k):
        """Training-side logloss of a k, measured on a subsample (a full self-prediction is O(n^2))."""
        S = self._train_sample()
        p, _ = self._vote_from_neighbors(*self._raw_kneighbors(S, min(k, len(self.X_train_))))
        y = self.y_train_[:len(S)]
        return _wlogloss(y, np.clip(p, 1e-6, 1 - 1e-6), np.ones(len(y)))

    def _raw_kneighbors(self, Z, k):
        k = max(1, min(k, len(self.X_train_)))
        out_d = np.empty((len(Z), k))
        out_i = np.empty((len(Z), k), dtype=int)
        for start in range(0, len(Z), self.block):
            chunk = Z[start:start + self.block]
            D = self._distances(chunk, self.X_train_)
            idx = np.argpartition(D, k - 1, axis=1)[:, :k]
            rows = np.arange(len(chunk))[:, None]
            idx = idx[rows, np.argsort(D[rows, idx], axis=1)]
            out_i[start:start + len(chunk)] = idx
            out_d[start:start + len(chunk)] = D[rows, idx]
        return out_d, out_i

    def _vote_from_neighbors(self, D, I):
        """Weighted vote given already-found neighbours (D distances, I positions)."""
        lab = self.y_train_[I]
        w = self.cw_[I].astype(float)                       # class/imbalance weight of each neighbour
        if self.weights == "distance":
            zero = D <= 1e-12                               # exact duplicates decide on their own
            w = w / np.where(zero, 1.0, D)
            if zero.any():
                w = np.where(zero, 1.0, w)                  # ... but only the duplicate rows count
        total = np.maximum(w.sum(axis=1), 1e-12)
        return (w * lab).sum(axis=1) / total, lab

    def _fit_platt(self, scores, y, cw):
        """1-D logistic regression P(y=1|s) = sigmoid(A*s/sd + B) by Newton-Raphson (Platt scaling)."""
        s = np.asarray(scores, dtype=float)
        y = np.asarray(y).astype(int)
        self._platt_sd = float(s.std()) or 1.0
        s = s / self._platt_sd
        A, B, ridge = 0.0, 0.0, 1e-3
        for _ in range(60):
            p = _sigmoid(A * s + B)
            g = cw * (p - y)
            h = cw * p * (1 - p) + ridge
            gA = float(np.sum(g * s)) + ridge * A
            gB = float(np.sum(g))
            hAA = float(np.sum(h * s * s)) + ridge
            hBB = float(np.sum(h)) + ridge
            hAB = float(np.sum(h * s))
            det = hAA * hBB - hAB * hAB
            if abs(det) < 1e-12:
                break
            dA = (hBB * gA - hAB * gB) / det
            dB = (hAA * gB - hAB * gA) / det
            A -= dA
            B -= dB
            if max(abs(dA), abs(dB)) < 1e-10:
                break
        self.platt_ = (float(A), float(B), float(self._platt_sd))

    # ---------------------------------------------------------------- predict
    def decision_function(self, X):
        """The raw vote share of the positive class (already in [0, 1]; positive = predicts default)."""
        t0 = time.time()
        self._check_fitted()
        p, _, _ = self._vote(X)
        self.predict_time_ = time.time() - t0
        return p

    def predict_proba(self, X):
        """Calibrated P(y=1|x): Platt scaling of the neighbour vote when it was fitted."""
        p = self.decision_function(X)
        if getattr(self, "platt_", None) is not None:
            A, B, sd = self.platt_
            p = _sigmoid(A * (np.asarray(p, dtype=float) / sd) + B)
        p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
        return np.column_stack([1.0 - p, p])

    def _check_fitted(self):
        if not getattr(self, "fitted_", False):
            raise RuntimeError("KNNClassifierScratch is not fitted yet - call fit(X, y) first")

    # ------------------------------------------------------- explain / importances
    def explain(self, x, top=5):
        """
        Why this answer? For k-NN the answer IS the neighbourhood: return the closest training rows
        with their labels, distances and (class) weights.
        """
        self._check_fitted()
        D, I = self.kneighbors(np.asarray(x, dtype=float).reshape(1, -1), top)
        nbrs = [(int(i), int(self.y_train_[i]), float(d), float(self.cw_[i]))
                for i, d in zip(I[0], D[0])]
        return {"vote": float(self.decision_function(np.asarray(x, dtype=float).reshape(1, -1))[0]),
                "p_default": float(self.predict_proba(np.asarray(x, dtype=float).reshape(1, -1))[0, 1]),
                "k": self.n_neighbors_, "neighbors": nbrs}

    def feature_importance_table(self, top=None, n_repeats=1, max_rows=300):
        """
        k-NN has no coefficients, so importance = PERMUTATION importance: shuffle one column of the
        validation rows, look at how much the ROC-AUC drops, and give the column that hurt most the
        biggest score. It needs the eval_set that was passed to fit(); without it every feature gets
        the same weight and the table says so instead of inventing numbers.
        """
        self._check_fitted()
        if self.feature_importances_ is None:
            if self._platt_eval is None:
                self.feature_importances_ = np.ones(self.n_features_in_) / self.n_features_in_
            else:
                Zv, yv = self._platt_eval
                m = min(len(Zv), int(max_rows))
                Zs, ys = Zv[:m], yv[:m]
                base = _auc(ys, self._vote_raw(Zs))
                imp = np.zeros(self.n_features_in_)
                rng = np.random.RandomState(self.random_state)
                for j in range(self.n_features_in_):
                    drop = 0.0
                    for _ in range(int(n_repeats)):
                        Zp = Zs.copy()
                        Zp[:, j] = Zp[rng.permutation(len(Zp)), j]      # shuffle this column
                        drop += max(0.0, base - _auc(ys, self._vote_raw(Zp)))
                    imp[j] = drop / max(1, int(n_repeats))
                self.feature_importances_ = imp / max(imp.sum(), 1e-12)
        vals = self.feature_importances_
        order = np.argsort(-vals)
        if top is not None:
            order = order[:top]
        return [(self.feature_names_[j], float(vals[j])) for j in order]

    def _vote_raw(self, Z):
        p, _ = self._vote_from_neighbors(*self._raw_kneighbors(Z, self.n_neighbors_))
        return p

    def coefficient_table(self, top=None):
        raise ValueError("k-NN has no coefficients (its model IS the stored data); use "
                         "feature_importance_table() for permutation importances instead")

    # ---------------------------------------------------------------- summaries
    def summary(self):
        self._check_fitted()
        ks = self.k_selection_
        if ks.get("metric"):
            line = (f"k-NN: k={self.n_neighbors_} chosen on validation by {ks['metric']} "
                    f"({ks['best_value']:.4f}) | neighbours voted with weights='{self.weights}' "
                    f"| metric={self.metric} | clip={self.clip}")
        else:
            line = (f"k-NN: k={self.n_neighbors_} (not tuned - no eval_set) | "
                    f"weights='{self.weights}' | metric={self.metric} | clip={self.clip}")
        return line + f"\n  stored {len(self.X_train_)} rows x {self.n_features_in_} features " \
                      f"(scaled={self.scale}) | fit {self.fit_time_:.2f}s"

    def get_state(self):
        self._check_fitted()
        return {"cls": "KNNClassifierScratch",
                "params": dict(n_neighbors=self.n_neighbors, weights=self.weights, metric=self.metric,
                               scale=self.scale, clip=self.clip, scale_pos_weight=self.scale_pos_weight,
                               tune_k=self.tune_k, k_grid=self.k_grid, tune_metric=self.tune_metric,
                               block=self.block, random_state=self.random_state),
                "arrays": {"X_train": self.X_train_, "y_train": self.y_train_, "cw": self.cw_,
                           "mean": self.mean_, "scale": self.scale_, "platt": self.platt_,
                           "importances": self.feature_importances_, "history": self.history_,
                           "k_selection": self.k_selection_},
                "meta": {"n_neighbors_": self.n_neighbors_, "feature_names": self.feature_names_,
                         "feature_importances_": self.feature_importances_,
                         "best_threshold": float(self.best_threshold_),
                         "n_features_in_": self.n_features_in_}}

    @classmethod
    def from_state(cls, s):
        m = cls(**s["params"])
        a = s["arrays"]
        m.X_train_, m.y_train_, m.cw_ = a["X_train"], a["y_train"], a["cw"]
        m.mean_, m.scale_, m.platt_ = a["mean"], a["scale"], a["platt"]
        m.feature_importances_ = a["importances"]
        m.history_, m.k_selection_ = a["history"], a["k_selection"]
        m.n_neighbors_, m.feature_names_ = s["meta"]["n_neighbors_"], s["meta"]["feature_names"]
        m.feature_importances_ = s["meta"]["feature_importances_"]
        m.best_threshold_ = s["meta"]["best_threshold"]
        m.n_features_in_ = s["meta"]["n_features_in_"]
        m.best_threshold_score_ = None
        m._platt_eval = None
        m.fitted_ = True
        m.fit_time_ = 0.0
        m.predict_time_ = None
        return m

    def save(self, path):
        import joblib
        joblib.dump({"model": "KNNClassifierScratch", "state": self.get_state()}, path)

    @classmethod
    def load(cls, path):
        import joblib
        return cls.from_state(joblib.load(path)["state"])


# alias so the class can be found under either name (the folder uses both styles)
KNNScratch = KNNClassifierScratch
