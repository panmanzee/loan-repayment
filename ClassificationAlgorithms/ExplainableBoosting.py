"""
Explainable Boosting Machine (EBM / "GA2M")  -  the BETTER, EXTRACURRICULAR classification model (built from scratch)
Classification task (target: not.fully.paid, 1 = defaulted).   Self-contained NumPy file (does not import Common/).

WHY THIS MODEL (and why it is not "just Gradient Boosting")
-----------------------------------------------------------
Gradient Boosting / XGBoost (taught in W5) grow DEEP trees on ALL features at once. They are accurate, but the result
is a black box: nobody can say "this applicant was refused because their utilisation is 92%". A bank is legally
required to give reasons (adverse-action notices). An EBM keeps most of the accuracy and is completely readable,
because the model is ADDITIVE - a "credit scorecard" whose point tables are LEARNED by boosting:

        log-odds(default | x)  =  b  +  f_1(x_1)  +  f_2(x_2)  +  ...  +  f_d(x_d)   ( + a few f_ij(x_i, x_j) pair terms )
        P(default | x)         =  sigmoid( log-odds )

Each f_j is a lookup table over the bins of ONE feature (a "shape function" you can draw as a curve). Logistic regression
is the special case where every f_j is a straight line; the EBM lets each curve bend where the data says so
(e.g. risk rises sharply above 80% utilisation but is flat below 40%).

HOW THE TABLES ARE LEARNED  (cyclic gradient boosting, one feature at a time)
-----------------------------------------------------------------------------
Start with b = weighted log-odds of the positive class and every f_j = 0. Then repeat for R rounds:
    for j = 1..d  (ROUND ROBIN - this is the key difference from Gradient Boosting, which picks the best feature):
        p_i = sigmoid(current log-odds)
        g_i = c_i (y_i - p_i)                    gradient (what the model still gets wrong)
        h_i = c_i p_i (1 - p_i)                  curvature
        bin-level sums:  G_k = sum_{i in bin k} g_i ,  H_k = sum_{i in bin k} h_i ,  N_k = rows in bin k
        grow a SMALL tree on the bins of feature j only (<= max_leaves leaves, splits between neighbouring bins):
            gain(split) = G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - G^2/(H+lam)      (Newton / XGBoost-style gain, min_samples_leaf enforced)
            leaf value  = G_leaf / (H_leaf + lam)
        f_j[bins in leaf]  +=  learning_rate * leaf value                      (a tiny step: lr = 0.02-0.05)
Because each tree sees only ONE feature, effects can never be mixed up between features, and the tiny learning rate with
many cycles makes the final curves smooth and order-independent. This is the Newton step of logistic regression applied
to one feature's bins - "boosted logistic regression with bins".

THE OTHER INGREDIENTS (all implemented below)
---------------------------------------------
* Quantile binning     : each feature is cut into <= max_bins equal-frequency bins (binary / few-valued features keep their values).
* Outer bagging        : n_outer_bags bootstrap resamples are boosted independently and their tables AVERAGED (variance reduction,
                         and the spread between bags gives an uncertainty band for every curve: `shape_function(..., with_std=True)`).
* Early stopping       : each bag stops when the validation (weighted) log-loss has not improved for `early_stopping_rounds` cycles.
* Pair interactions    : optional `n_interactions` > 0 (GA2M): after the main effects, every pair of features is scored with a
                         fast 2-D histogram of the remaining gradient, the strongest pairs get a 2-D table (coarse bins) that is
                         also boosted with Newton steps. Still readable as a heat-map.
* Centering            : every f_j is shifted to weighted mean 0 and the shift is moved into b, so |f_j| directly measures importance.
* Linear backbone      : `linear_base=True` (default) first fits a regularised logistic regression (Newton / IRLS) and boosts only
                         the CORRECTION on top of it. The model is therefore never worse than logistic regression on the signal that
                         is linear, and the bins only have to learn where the data bends (thresholds, saturation, U-shapes). Every
                         feature's drawn curve = its linear part + its learned bin correction.
* Class imbalance      : defaulters get weight sqrt((n - n_pos) / n_pos) (project convention) via `scale_pos_weight="auto"`.

OUTPUTS THAT MAKE IT "EXPLAINABLE"
----------------------------------
    shape_function(j)   -> bin centres and log-odds contribution of feature j (the learned curve)
    explain(x)          -> contribution of every feature for ONE borrower (sums to the log-odds)
    feature_importances_-> weighted mean |f_j| (share of total)
    export_text()       -> the scorecard in words

Example
-------
    ebm = ExplainableBoostingScratch(n_rounds=600, learning_rate=0.03, n_outer_bags=8, n_interactions=0)
    ebm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], feature_names=names, early_stopping_rounds=40)
    proba = ebm.predict_proba(X_te)[:, 1]
    print(ebm.export_text(top=6));  print(ebm.explain(X_te[0]))
"""
import time

import numpy as np


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def sqrt_scale_pos_weight(y):
    y = np.asarray(y, dtype=int).ravel(); n_pos = int(y.sum())
    return 1.0 if n_pos in (0, len(y)) else float(np.sqrt((len(y) - n_pos) / n_pos))


def _auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]; i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _wlogloss(y, p, w):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=w))


def _fit_linear_backbone(Z, y, cw, l2=1e-3, iters=25):
    """Weighted L2-regularised logistic regression by Newton-Raphson (IRLS). Z is standardised. Returns (beta, intercept)."""
    n, d = Z.shape
    A = np.column_stack([Z, np.ones(n)]); th = np.zeros(d + 1)
    th[-1] = np.log(np.sum(cw * y) / np.sum(cw * (1 - y)))
    lam = l2 * cw.sum(); R = lam * np.eye(d + 1); R[-1, -1] = 0.0           # do not penalise the intercept
    for _ in range(iters):
        p = _sigmoid(A @ th); g = A.T @ (cw * (p - y)) + R @ th
        H = (A * (cw * p * (1 - p))[:, None]).T @ A + R + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(H, g); th -= step
        if np.abs(step).max() < 1e-9:
            break
    return th[:-1], float(th[-1])


def _grow_bin_tree(G, H, N, max_leaves, min_leaf, lam, min_gain=1e-9):
    """
    Greedy tree on ONE feature's bin histogram. Returns [(start, end, leaf_value)] covering all bins [0, nb).
    Splits are only allowed BETWEEN neighbouring bins, so the tree is a piecewise-constant step function.
    """
    nb = len(G)
    segs = [(0, nb)]
    for _ in range(max_leaves - 1):
        best = (min_gain, None, None)
        for si, (a, b) in enumerate(segs):
            if b - a < 2:
                continue
            g, h, n = G[a:b], H[a:b], N[a:b]
            cg, ch, cn = np.cumsum(g)[:-1], np.cumsum(h)[:-1], np.cumsum(n)[:-1]
            Gt, Ht, Nt = g.sum(), h.sum(), n.sum()
            ok = (cn >= min_leaf) & (Nt - cn >= min_leaf)
            if not ok.any():
                continue
            gain = cg ** 2 / (ch + lam) + (Gt - cg) ** 2 / (Ht - ch + lam) - Gt ** 2 / (Ht + lam)
            gain = np.where(ok, gain, -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > best[0]:
                best = (float(gain[k]), si, a + k + 1)
        if best[1] is None:
            break
        a, b = segs[best[1]]
        segs[best[1]:best[1] + 1] = [(a, best[2]), (best[2], b)]
    return [(a, b, float(G[a:b].sum() / (H[a:b].sum() + lam))) for a, b in segs]


class ExplainableBoostingScratch:
    def __init__(self, n_rounds=600, learning_rate=0.03, max_bins=32, max_leaves=3, min_samples_leaf=30, reg_lambda=1.0,
                 n_outer_bags=8, n_interactions=0, interaction_bins=6, interaction_learning_rate=0.02,
                 interaction_min_cell=25, interaction_rounds=150, linear_base=True, linear_l2=1e-3,
                 scale_pos_weight="auto", random_state=42):
        self.n_rounds, self.learning_rate, self.max_bins, self.max_leaves = n_rounds, learning_rate, max_bins, max_leaves
        self.min_samples_leaf, self.reg_lambda, self.n_outer_bags = min_samples_leaf, reg_lambda, n_outer_bags
        self.n_interactions, self.interaction_bins, self.interaction_learning_rate = n_interactions, interaction_bins, interaction_learning_rate
        self.interaction_min_cell, self.interaction_rounds = interaction_min_cell, interaction_rounds
        self.linear_base, self.linear_l2 = linear_base, linear_l2
        self.scale_pos_weight, self.random_state = scale_pos_weight, random_state

    # ------------------------------------------------------------ binning
    def _make_bins(self, X, nbins):
        edges = []
        for j in range(X.shape[1]):
            u = np.unique(X[:, j])
            if len(u) <= nbins:
                e = (u[:-1] + u[1:]) / 2.0
            else:
                e = np.unique(np.quantile(X[:, j], np.linspace(0, 1, nbins + 1)[1:-1]))
            edges.append(e)
        return edges

    @staticmethod
    def _bin(X, edges):
        return np.column_stack([np.searchsorted(edges[j], X[:, j], side="right") for j in range(X.shape[1])]).astype(np.int32)

    def _zlin(self, X):
        return (np.asarray(X, dtype=np.float64) - self.lin_mean_) / self.lin_scale_

    # ---------------------------------------------------------------- fit
    def fit(self, X, y, eval_set=None, feature_names=None, early_stopping_rounds=40, sample_weight=None, verbose=False, **_ignored):
        t0 = time.time()
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y).astype(float).ravel(); n, d = X.shape
        self.n_features_ = d
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(d)]
        spw = sqrt_scale_pos_weight(y) if self.scale_pos_weight == "auto" else float(self.scale_pos_weight)
        self.scale_pos_weight_ = spw
        cw = np.where(y == 1, spw, 1.0)
        if sample_weight is not None:
            cw = cw * np.asarray(sample_weight, dtype=float)
        self.edges_ = self._make_bins(X, self.max_bins)
        B = self._bin(X, self.edges_); nbins = [len(e) + 1 for e in self.edges_]
        self.n_bins_ = nbins
        Bv = yv = cwv = None
        if eval_set:
            Bv = self._bin(np.asarray(eval_set[0][0], dtype=np.float64), self.edges_)
            yv = np.asarray(eval_set[0][1]).astype(float); cwv = np.where(yv == 1, spw, 1.0)
        rng = np.random.RandomState(self.random_state)
        b0 = float(np.log(np.sum(cw * y) / np.sum(cw * (1 - y))))
        self.lin_mean_ = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1.0; self.lin_scale_ = sd
        self.lin_coef_ = np.zeros(d)
        F0 = np.full(n, b0); F0v = np.full(len(yv), b0) if Bv is not None else None
        if self.linear_base:
            self.lin_coef_, b0 = _fit_linear_backbone(self._zlin(X), y, cw, self.linear_l2)
            F0 = b0 + self._zlin(X) @ self.lin_coef_
            if Bv is not None:
                F0v = b0 + self._zlin(np.asarray(eval_set[0][0], dtype=np.float64)) @ self.lin_coef_
        bags_tables, bags_hist, bags_rounds = [], [], []
        nbag = max(1, int(self.n_outer_bags))
        for bag in range(nbag):
            counts = np.ones(n) if nbag == 1 else np.bincount(rng.randint(0, n, n), minlength=n).astype(float)
            w = cw * counts
            tables = [np.zeros(k) for k in nbins]
            F = F0.copy(); Fv = F0v.copy() if Bv is not None else None
            best = (np.inf, [t.copy() for t in tables], 0); bad = 0
            hist = {"train_loss": [], "val_loss": [], "val_auc": [], "train_logloss": [], "val_logloss": []}
            for r in range(1, self.n_rounds + 1):
                for j in range(d):
                    p = _sigmoid(F)
                    g = w * (y - p); h = w * p * (1 - p)
                    G = np.bincount(B[:, j], weights=g, minlength=nbins[j]); H = np.bincount(B[:, j], weights=h, minlength=nbins[j])
                    N = np.bincount(B[:, j], weights=counts, minlength=nbins[j])
                    delta = np.zeros(nbins[j])
                    for a, b, val in _grow_bin_tree(G, H, N, self.max_leaves, self.min_samples_leaf, self.reg_lambda):
                        delta[a:b] = self.learning_rate * val
                    tables[j] += delta; F += delta[B[:, j]]
                    if Fv is not None:
                        Fv += delta[Bv[:, j]]
                hist["train_loss"].append(_wlogloss(y, _sigmoid(F), w)); hist["train_logloss"].append(_wlogloss(y, _sigmoid(F), np.ones(n)))
                if Fv is not None:
                    pv = _sigmoid(Fv); vl = _wlogloss(yv, pv, cwv)
                    hist["val_loss"].append(vl); hist["val_auc"].append(_auc(yv, pv)); hist["val_logloss"].append(_wlogloss(yv, pv, np.ones(len(yv))))
                    if vl < best[0] - 1e-7:
                        best = (vl, [t.copy() for t in tables], r); bad = 0
                    else:
                        bad += 1
                    if early_stopping_rounds and bad >= early_stopping_rounds:
                        break
            if Fv is not None:
                tables = best[1]
            bags_tables.append(tables); bags_hist.append(hist); bags_rounds.append(best[2] if Fv is not None else r)
            if verbose:
                print(f"  bag {bag + 1}/{nbag}: {len(hist['train_loss'])} cycles, best cycle {bags_rounds[-1]}")
        self.bag_tables_ = bags_tables; self.bag_best_rounds_ = bags_rounds
        self.tables_ = [np.mean([bt[j] for bt in bags_tables], axis=0) for j in range(d)]
        self.table_std_ = [np.std([bt[j] for bt in bags_tables], axis=0) for j in range(d)]
        self.intercept_ = b0
        self.bin_weight_ = [np.bincount(B[:, j], weights=cw, minlength=nbins[j]) for j in range(d)]
        # centre every curve (weighted mean 0 over the training rows) and push the shift into the intercept
        for j in range(d):
            m = float(np.sum(self.tables_[j] * self.bin_weight_[j]) / self.bin_weight_[j].sum())
            self.tables_[j] = self.tables_[j] - m; self.intercept_ += m
        # bag-averaged learning curves (carry the last value forward for bags that stopped earlier)
        L = max(len(h["train_loss"]) for h in bags_hist)
        def avg(key):
            if not bags_hist[0][key]:
                return []
            return list(np.mean([np.r_[h[key], np.full(L - len(h[key]), h[key][-1])] for h in bags_hist], axis=0))
        self.history_ = {"epoch": list(range(1, L + 1)), "train_loss": avg("train_loss"), "val_loss": avg("val_loss"), "val_auc": avg("val_auc"),
                         "train_logloss": avg("train_logloss"), "val_logloss": avg("val_logloss")}
        self.best_round_ = int(np.mean(bags_rounds))
        self.pairs_, self.pair_tables_, self.pair_edges_ = [], [], {}
        if self.n_interactions > 0:
            self._fit_interactions(X, y, cw, B, eval_set, early_stopping_rounds, verbose)
        self._refresh_importance(X, cw)
        self.fit_time_ = time.time() - t0
        self.best_threshold_ = 0.5
        return self

    # ------------------------------------------------- pairwise interactions
    def _fit_interactions(self, X, y, cw, B, eval_set, esr, verbose):
        n, d = X.shape; K = self.interaction_bins; lam = self.reg_lambda
        edges_c = self._make_bins(X, K); Bc = self._bin(X, edges_c)
        self.pair_edges_ = {j: edges_c[j] for j in range(d)}
        F = self._raw_from_bins(B) + self._zlin(X) @ self.lin_coef_
        p = _sigmoid(F); g = cw * (y - p); h = cw * p * (1 - p)
        # FAST-style screening: gain of a 2-D table on the gradient that the main effects left over
        scores = []
        corr = np.corrcoef(X, rowvar=False)
        for i in range(d):
            for j in range(i + 1, d):
                if len(edges_c[i]) == 0 or len(edges_c[j]) == 0:
                    continue
                if np.isfinite(corr[i, j]) and abs(corr[i, j]) > 0.98:       # near-duplicate columns carry no real interaction
                    continue
                idx = Bc[:, i] * (len(edges_c[j]) + 1) + Bc[:, j]; m = (len(edges_c[i]) + 1) * (len(edges_c[j]) + 1)
                G = np.bincount(idx, weights=g, minlength=m); H = np.bincount(idx, weights=h, minlength=m)
                Gi = np.bincount(Bc[:, i], weights=g, minlength=len(edges_c[i]) + 1); Hi = np.bincount(Bc[:, i], weights=h, minlength=len(edges_c[i]) + 1)
                Gj = np.bincount(Bc[:, j], weights=g, minlength=len(edges_c[j]) + 1); Hj = np.bincount(Bc[:, j], weights=h, minlength=len(edges_c[j]) + 1)
                gain = (G ** 2 / (H + lam)).sum() - (Gi ** 2 / (Hi + lam)).sum() - (Gj ** 2 / (Hj + lam)).sum()
                scores.append((gain, i, j))
        scores.sort(reverse=True)
        self.pairs_ = [(i, j) for _, i, j in scores[:self.n_interactions]]
        self.pair_scores_ = {(i, j): float(s) for s, i, j in scores[:self.n_interactions]}
        shapes = [((len(edges_c[i]) + 1), (len(edges_c[j]) + 1)) for i, j in self.pairs_]
        self.pair_tables_ = [np.zeros(s) for s in shapes]
        idxs = [Bc[:, i] * s[1] + Bc[:, j] for (i, j), s in zip(self.pairs_, shapes)]
        Bcv = yv = cwv = None
        if eval_set:
            Bcv = self._bin(np.asarray(eval_set[0][0], dtype=np.float64), edges_c); yv = np.asarray(eval_set[0][1]).astype(float)
            cwv = np.where(yv == 1, self.scale_pos_weight_, 1.0)
        best = (np.inf, [t.copy() for t in self.pair_tables_], 0); bad = 0
        Bv_main = self._bin(np.asarray(eval_set[0][0], dtype=np.float64), self.edges_) if eval_set else None
        base_v = (self._raw_from_bins(Bv_main) + self._zlin(np.asarray(eval_set[0][0], dtype=np.float64)) @ self.lin_coef_) if eval_set else None
        for r in range(1, self.interaction_rounds + 1):
            for q, ((i, j), s) in enumerate(zip(self.pairs_, shapes)):
                p = _sigmoid(F); g = cw * (y - p); h = cw * p * (1 - p)
                m = s[0] * s[1]
                G = np.bincount(idxs[q], weights=g, minlength=m); H = np.bincount(idxs[q], weights=h, minlength=m)
                Nc = np.bincount(idxs[q], minlength=m)
                delta = np.where(Nc >= self.interaction_min_cell, self.interaction_learning_rate * G / (H + lam), 0.0)
                self.pair_tables_[q] = self.pair_tables_[q] + delta.reshape(s); F += delta[idxs[q]]
            if eval_set:
                Fv = base_v.copy()
                for (i, j), s, t in zip(self.pairs_, shapes, self.pair_tables_):
                    Fv += t.ravel()[Bcv[:, i] * s[1] + Bcv[:, j]]
                vl = _wlogloss(yv, _sigmoid(Fv), cwv)
                if vl < best[0] - 1e-7:
                    best = (vl, [t.copy() for t in self.pair_tables_], r); bad = 0
                else:
                    bad += 1
                if esr and bad >= max(10, esr // 2):
                    break
        if eval_set:
            self.pair_tables_ = best[1]
        self.interaction_best_round_ = best[2] if eval_set else self.interaction_rounds
        # centre each pair table too
        for q, ((i, j), s) in enumerate(zip(self.pairs_, shapes)):
            m = float(np.sum(self.pair_tables_[q].ravel()[idxs[q]] * cw) / cw.sum())
            self.pair_tables_[q] = self.pair_tables_[q] - m; self.intercept_ += m
        if verbose:
            print("  interaction pairs:", [(self.feature_names_[i], self.feature_names_[j]) for i, j in self.pairs_])

    # ------------------------------------------------------------ prediction
    def _raw_from_bins(self, B):
        F = np.full(B.shape[0], self.intercept_)
        for j in range(B.shape[1]):
            F += self.tables_[j][B[:, j]]
        return F

    def contributions(self, X):
        """(n, d) matrix of main-effect contributions + (n, n_pairs) pair contributions."""
        X = np.asarray(X, dtype=np.float64); B = self._bin(X, self.edges_)
        C = np.column_stack([self.tables_[j][B[:, j]] for j in range(self.n_features_)]) + self._zlin(X) * self.lin_coef_[None, :]
        P = None
        if self.pairs_:
            Bc = self._bin(X, [self.pair_edges_[j] for j in range(self.n_features_)])
            P = np.column_stack([t[Bc[:, i], Bc[:, j]] for (i, j), t in zip(self.pairs_, self.pair_tables_)])
        return C, P

    def decision_function(self, X):
        C, P = self.contributions(X)
        return self.intercept_ + C.sum(axis=1) + (P.sum(axis=1) if P is not None else 0.0)

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=None):
        t = self.best_threshold_ if threshold is None else threshold
        return (self.predict_proba(X)[:, 1] >= t).astype(int)

    def pick_threshold(self, X_val, y_val, metric="f1"):
        p = self.predict_proba(X_val)[:, 1]; y = np.asarray(y_val).astype(int)
        best_t, best_s = 0.5, -1.0
        for t in np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97))):
            pr = (p >= t).astype(int)
            tp = ((pr == 1) & (y == 1)).sum(); fp = ((pr == 1) & (y == 0)).sum()
            fn = ((pr == 0) & (y == 1)).sum(); tn = ((pr == 0) & (y == 0)).sum()
            s = 2 * tp / max(2 * tp + fp + fn, 1) if metric == "f1" else 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
            if s > best_s:
                best_t, best_s = float(t), float(s)
        self.best_threshold_, self.best_threshold_score_ = best_t, best_s
        return best_t


    # --- plug-in names used by run_classification.py (unweighted log-loss curves) ---
    @property
    def train_loss(self):
        return self.history_.get("train_logloss") if getattr(self, "history_", None) else None

    @property
    def evals_result_(self):
        h = getattr(self, "history_", None) or {}
        return {"validation": {"logloss": h.get("val_logloss", [])}}

    # --------------------------------------------------------- explanations
    def _refresh_importance(self, X, cw):
        C, _ = self.contributions(X)
        imp = (cw[:, None] * np.abs(C - np.average(C, axis=0, weights=cw))).sum(axis=0) / cw.sum()
        self.main_importances_ = imp
        self.feature_importances_ = imp / max(imp.sum(), 1e-12)

    def shape_function(self, j, with_std=False):
        """(bin_centres, log-odds contribution) of feature j - the learned curve. Centre = representative value of each bin."""
        e = self.edges_[j]
        if len(e) == 0:
            return np.array([0.0]), self.tables_[j]
        lo = e[0] - (e[1] - e[0] if len(e) > 1 else 1.0); hi = e[-1] + (e[-1] - e[-2] if len(e) > 1 else 1.0)
        bounds = np.r_[lo, e, hi]; centres = (bounds[:-1] + bounds[1:]) / 2.0
        curve = self.tables_[j] + self.lin_coef_[j] * (centres - self.lin_mean_[j]) / self.lin_scale_[j]       # linear part + learned correction
        curve = curve - np.sum(curve * self.bin_weight_[j]) / self.bin_weight_[j].sum()
        return (centres, curve, self.table_std_[j]) if with_std else (centres, curve)

    def explain(self, x, top=6):
        """Contribution of each feature to ONE borrower's log-odds (sorted by size). Intercept + contributions = log-odds."""
        C, P = self.contributions(np.atleast_2d(x))
        rows = [(self.feature_names_[j], float(C[0, j])) for j in range(self.n_features_)]
        if P is not None:
            rows += [(f"{self.feature_names_[i]} x {self.feature_names_[j]}", float(P[0, q])) for q, (i, j) in enumerate(self.pairs_)]
        rows.sort(key=lambda r: -abs(r[1]))
        return {"intercept": float(self.intercept_), "contributions": rows[:top],
                "log_odds": float(self.intercept_ + C.sum() + (P.sum() if P is not None else 0.0))}

    def feature_importance_table(self, top=None):
        order = np.argsort(-self.feature_importances_)
        rows = [(self.feature_names_[j], float(self.feature_importances_[j])) for j in order]
        return rows[:top] if top else rows

    def export_text(self, top=6):
        lines = [f"log-odds(default) = {self.intercept_:+.3f} + sum_j f_j(x_j)" + (f" + {len(self.pairs_)} pair term(s)" if self.pairs_ else ""), ""]
        for j in np.argsort(-self.feature_importances_)[:top]:
            c, t = self.shape_function(j)
            lo, hi = int(np.argmin(t)), int(np.argmax(t))
            lines.append(f"{self.feature_names_[j]:<24} share {self.feature_importances_[j]:.1%}   lowest risk near {c[lo]:.4g} ({t[lo]:+.3f})   highest risk near {c[hi]:.4g} ({t[hi]:+.3f})")
        return "\n".join(lines)

    # ------------------------------------------------------------ save / load
    def get_state(self):
        return {"cls": "ExplainableBoostingScratch",
                "params": dict(n_rounds=self.n_rounds, learning_rate=self.learning_rate, max_bins=self.max_bins, max_leaves=self.max_leaves,
                               min_samples_leaf=self.min_samples_leaf, reg_lambda=self.reg_lambda, n_outer_bags=self.n_outer_bags,
                               n_interactions=self.n_interactions, interaction_bins=self.interaction_bins,
                               interaction_learning_rate=self.interaction_learning_rate, interaction_min_cell=self.interaction_min_cell,
                               interaction_rounds=self.interaction_rounds, linear_base=self.linear_base, linear_l2=self.linear_l2, scale_pos_weight=self.scale_pos_weight, random_state=self.random_state),
                "feature_names": self.feature_names_, "edges": self.edges_, "tables": self.tables_, "table_std": self.table_std_,
                "intercept": float(self.intercept_), "bin_weight": self.bin_weight_, "pairs": self.pairs_, "pair_tables": self.pair_tables_,
                "pair_edges": self.pair_edges_, "importances": self.feature_importances_, "best_threshold": float(self.best_threshold_),
                "history": self.history_, "scale_pos_weight_": float(self.scale_pos_weight_),
                "lin_mean": self.lin_mean_, "lin_scale": self.lin_scale_, "lin_coef": self.lin_coef_}

    @classmethod
    def from_state(cls, s):
        m = cls(**s["params"]); m.feature_names_ = s["feature_names"]; m.n_features_ = len(m.feature_names_)
        m.edges_, m.tables_, m.table_std_, m.intercept_ = s["edges"], s["tables"], s["table_std"], s["intercept"]
        m.bin_weight_, m.pairs_, m.pair_tables_, m.pair_edges_ = s["bin_weight"], s["pairs"], s["pair_tables"], s["pair_edges"]
        m.feature_importances_, m.best_threshold_, m.history_ = s["importances"], s["best_threshold"], s["history"]
        m.lin_mean_, m.lin_scale_, m.lin_coef_ = s["lin_mean"], s["lin_scale"], s["lin_coef"]
        m.scale_pos_weight_ = s["scale_pos_weight_"]; m.n_bins_ = [len(e) + 1 for e in m.edges_]
        return m

    def save(self, path):
        import joblib
        joblib.dump({"model": "ExplainableBoostingScratch", "state": self.get_state()}, path)

    @classmethod
    def load(cls, path):
        import joblib
        return cls.from_state(joblib.load(path)["state"])
