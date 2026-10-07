"""
Perceptron  &  Single-Layer Perceptron (SLP)  (from scratch) - Classification task (target: not.fully.paid)

Everything is written from scratch with NumPy. The file is self-contained (it does not import Common/).

1) PerceptronScratch  -  Rosenblatt's perceptron (1958)
----------------------------------------------------------
One artificial neuron with a STEP activation:

        s = w . x + b                      <- weighted sum of the (standardised) features
        y_hat = +1 if s > 0 else -1        <- hard decision, no probability at all

Learning is "mistake driven": look at one borrower at a time, and ONLY when the neuron is wrong move the
separating hyperplane a little towards that borrower:

        if t_i * s_i <= 0 :   w <- w + eta * c_i * t_i * x_i ;   b <- b + eta * c_i * t_i          (t = +1 default / -1 paid)

c_i is the sample weight (defaulters count sqrt((n-n_pos)/n_pos) times more - the project's imbalance convention).
Perceptron convergence theorem: if the classes can be separated by a straight line, this reaches zero mistakes in
a finite number of updates. LOAN DATA IS NOT LINEARLY SEPARABLE (the classes overlap heavily), so the classic
algorithm never settles and keeps jumping around. Two standard repairs are implemented:

    variant="pocket"   : keep ("put in the pocket") the weights of the best epoch seen so far (lowest weighted error)
    variant="averaged" : return the AVERAGE of the weights over every step (Freund & Schapire) - very stable

A step neuron has no probability, so `predict_proba` squashes the score with a 1-D logistic curve fitted on the
training scores (Platt scaling, solved by Newton's method below). ROC-AUC only needs the raw score
(`decision_function`) and is unaffected by that squashing.

2) SingleLayerPerceptronScratch (SLP)  -  one neuron with a SIGMOID activation, trained by gradient descent
-----------------------------------------------------------------------------------------------------
Replace the step by the smooth sigmoid, so the neuron outputs a probability and has a gradient:

        p_i  = sigmoid(w . x_i + b)
        loss = -(1/W) sum_i c_i [ y_i log p_i + (1-y_i) log(1-p_i) ]  +  (lambda/2) ||w||^2        (weighted binary cross-entropy)
        dloss/dz_i = c_i (p_i - y_i) / W   ->   grad_w = X^T dz + lambda w ,  grad_b = sum dz        (the "delta rule")

Mini-batch SGD with momentum, learning-rate decay, early stopping on the validation loss and automatic
restoring of the best epoch. HONEST NOTE: a single sigmoid neuron trained on cross-entropy IS logistic regression
(only the optimiser differs); the SLP is the building block of the MLP in MLP.py - add hidden layers and you
get the multi-layer network.

Both classes standardise X themselves (fitted on training rows only), keep `history_` for the loss / performance
curves, expose `feature_importances_` (|weight| per 1 SD) and have save() / load().

Example
-------
    slp = SingleLayerPerceptronScratch(learning_rate=0.05, n_epochs=200, l2=1e-3)
    slp.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], feature_names=names, early_stopping_rounds=20)
    proba = slp.predict_proba(X_te)[:, 1]
"""
import time

import numpy as np


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def sqrt_scale_pos_weight(y):
    """Project-wide imbalance weight for the positive rows: sqrt((n - n_pos) / n_pos)  (~2.29 here)."""
    y = np.asarray(y, dtype=int).ravel(); n_pos = int(y.sum())
    return 1.0 if n_pos in (0, len(y)) else float(np.sqrt((len(y) - n_pos) / n_pos))


def _auc(y, s):
    """ROC-AUC by the rank-sum (Mann-Whitney U) identity, ties get average rank."""
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]; i = 0
    while i < len(ss):                                   # average the ranks inside ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _weights(y, scale_pos_weight, sample_weight):
    y = np.asarray(y).astype(int)
    spw = sqrt_scale_pos_weight(y) if scale_pos_weight == "auto" else float(scale_pos_weight)
    w = np.where(y == 1, spw, 1.0)
    if sample_weight is not None:
        w = w * np.asarray(sample_weight, dtype=float)
    return w, spw


def _wlogloss(y, p, w):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=w))


class _Base:
    """Shared plumbing: standardisation, thresholds, importances, persistence."""

    def _fit_scaler(self, X):
        self.mean_ = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1.0; self.scale_ = sd

    def _z(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_

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

    def coefficient_table(self, top=None):
        order = np.argsort(-np.abs(self.coef_))
        rows = [{"feature": self.feature_names_[j], "weight_per_1SD": float(self.coef_[j])} for j in order]
        return rows[:top] if top else rows

    def feature_importance_table(self, top=None):
        rows = [(r["feature"], abs(r["weight_per_1SD"])) for r in self.coefficient_table()]
        return rows[:top] if top else rows

    def save(self, path):
        import joblib
        joblib.dump({"model": type(self).__name__, "state": self.get_state()}, path)

    @classmethod
    def load(cls, path):
        import joblib
        return cls.from_state(joblib.load(path)["state"])


# =============================================================================
# 1) classic perceptron (step activation)
# =============================================================================
class PerceptronScratch(_Base):
    def __init__(self, learning_rate=0.1, n_epochs=60, variant="averaged", scale_pos_weight="auto",
                 shuffle=True, random_state=42):
        if variant not in ("classic", "pocket", "averaged"):
            raise ValueError("variant must be 'classic', 'pocket' or 'averaged'")
        self.learning_rate, self.n_epochs, self.variant = learning_rate, n_epochs, variant
        self.scale_pos_weight, self.shuffle, self.random_state = scale_pos_weight, shuffle, random_state

    def _score(self, Z, w, b):
        return Z @ w + b

    def fit(self, X, y, eval_set=None, feature_names=None, sample_weight=None, verbose=False, **_ignored):
        t0 = time.time()
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y).astype(int).ravel()
        self._fit_scaler(X); Z = self._z(X); n, d = Z.shape
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(d)]
        cw, self.scale_pos_weight_ = _weights(y, self.scale_pos_weight, sample_weight)
        t = 2 * y - 1
        rng = np.random.RandomState(self.random_state)
        w = np.zeros(d); b = 0.0
        w_sum = np.zeros(d); b_sum = 0.0; steps = 0                      # for the averaged perceptron
        best = (np.inf, w.copy(), b)                                      # for the pocket
        hist = {k: [] for k in ("epoch", "mistakes", "train_error", "train_accuracy", "train_auc", "val_accuracy", "val_auc")}
        Zv = yv = None
        if eval_set:
            Zv, yv = self._z(eval_set[0][0]), np.asarray(eval_set[0][1]).astype(int)
        for ep in range(1, self.n_epochs + 1):
            order = rng.permutation(n) if self.shuffle else np.arange(n)
            mistakes = 0
            for i in order:
                if t[i] * (Z[i] @ w + b) <= 0:                            # wrong (or on the line) -> update
                    step = self.learning_rate * cw[i] * t[i]
                    w += step * Z[i]; b += step; mistakes += 1
                w_sum += w; b_sum += b; steps += 1
            s_tr = Z @ w + b
            werr = float(np.sum(cw * ((s_tr > 0).astype(int) != y)) / cw.sum())
            hist["epoch"].append(ep); hist["mistakes"].append(mistakes); hist["train_error"].append(werr)
            hist["train_accuracy"].append(float(((s_tr > 0).astype(int) == y).mean())); hist["train_auc"].append(_auc(y, s_tr))
            if Zv is not None:
                sv = Zv @ w + b
                hist["val_accuracy"].append(float(((sv > 0).astype(int) == yv).mean())); hist["val_auc"].append(_auc(yv, sv))
            if werr < best[0]:
                best = (werr, w.copy(), b)
            if verbose and ep % 10 == 0:
                print(f"  epoch {ep:>3} | mistakes {mistakes:>5} | weighted train error {werr:.4f}")
            if mistakes == 0:                                             # linearly separable -> converged
                self.converged_ = True; break
        else:
            self.converged_ = False
        if self.variant == "averaged":
            w_f, b_f = w_sum / steps, b_sum / steps
        elif self.variant == "pocket":
            w_f, b_f = best[1], best[2]
        else:
            w_f, b_f = w, b
        self.coef_, self.intercept_ = w_f.copy(), float(b_f)
        self.history_ = hist; self.n_epochs_run_ = len(hist["epoch"]); self.fit_time_ = time.time() - t0
        self._fit_platt(Z, y, cw)
        self.feature_importances_ = np.abs(self.coef_) / max(np.abs(self.coef_).sum(), 1e-12)
        self.best_threshold_ = 0.5
        return self

    def _fit_platt(self, Z, y, cw):
        """
        P(y=1|s) = sigmoid(A*s + B), fitted by Newton-Raphson on the weighted log-loss (1-D logistic regression).
        The raw score is first divided by its own spread (a perceptron's weight scale is arbitrary) and A gets a
        tiny ridge so a score with NO signal cannot send A to infinity.
        """
        s = Z @ self.coef_ + self.intercept_
        self._platt_sd = float(s.std()) or 1.0
        s = s / self._platt_sd
        A, B, ridge = 0.0, 0.0, 1e-3
        for _ in range(60):
            p = _sigmoid(A * s + B); g = cw * (p - y); h = cw * p * (1 - p) + 1e-12
            gr = np.array([np.sum(g * s) + ridge * A, np.sum(g)])
            H = np.array([[np.sum(h * s * s) + ridge, np.sum(h * s)], [np.sum(h * s), np.sum(h)]]) + 1e-9 * np.eye(2)
            step = np.linalg.solve(H, gr); A, B = A - step[0], B - step[1]
            if np.abs(step).max() < 1e-9:
                break
        self.platt_ = (float(A), float(B), self._platt_sd)

    def decision_function(self, X):
        return self._z(X) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p = _sigmoid(self.platt_[0] * self.decision_function(X) / self.platt_[2] + self.platt_[1])
        return np.column_stack([1 - p, p])

    def predict_step(self, X):
        """The pure perceptron decision: sign(w.x + b)  ->  1 if > 0 else 0 (no threshold, no probability)."""
        return (self.decision_function(X) > 0).astype(int)

    def get_state(self):
        return {"cls": "PerceptronScratch", "params": dict(learning_rate=self.learning_rate, n_epochs=self.n_epochs, variant=self.variant,
                                                           scale_pos_weight=self.scale_pos_weight, shuffle=self.shuffle, random_state=self.random_state),
                "mean": self.mean_, "scale": self.scale_, "coef": self.coef_, "intercept": self.intercept_, "platt": list(self.platt_),
                "feature_names": self.feature_names_, "best_threshold": float(self.best_threshold_), "history": self.history_}

    @classmethod
    def from_state(cls, s):
        m = cls(**s["params"]); m.mean_, m.scale_, m.coef_, m.intercept_ = s["mean"], s["scale"], s["coef"], s["intercept"]
        m.platt_ = tuple(s["platt"]); m.feature_names_ = s["feature_names"]; m.best_threshold_ = s["best_threshold"]
        m.history_ = s["history"]; m.feature_importances_ = np.abs(m.coef_) / max(np.abs(m.coef_).sum(), 1e-12)
        return m


# =============================================================================
# 2) single-layer perceptron (sigmoid activation, gradient descent)
# =============================================================================
class SingleLayerPerceptronScratch(_Base):
    def __init__(self, learning_rate=0.05, n_epochs=200, batch_size=128, momentum=0.9, l2=1e-3, lr_decay=0.01,
                 scale_pos_weight="auto", random_state=42):
        self.learning_rate, self.n_epochs, self.batch_size, self.momentum = learning_rate, n_epochs, batch_size, momentum
        self.l2, self.lr_decay, self.scale_pos_weight, self.random_state = l2, lr_decay, scale_pos_weight, random_state

    def fit(self, X, y, eval_set=None, feature_names=None, early_stopping_rounds=20, sample_weight=None, verbose=False, **_ignored):
        t0 = time.time()
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y).astype(int).ravel()
        self._fit_scaler(X); Z = self._z(X); n, d = Z.shape
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(d)]
        cw, self.scale_pos_weight_ = _weights(y, self.scale_pos_weight, sample_weight)
        rng = np.random.RandomState(self.random_state)
        w = rng.normal(0, 0.01, d); b = 0.0; vw = np.zeros(d); vb = 0.0
        Zv = yv = cwv = None
        if eval_set:
            Zv, yv = self._z(eval_set[0][0]), np.asarray(eval_set[0][1]).astype(int)
            cwv = np.where(yv == 1, self.scale_pos_weight_, 1.0)
        hist = {k: [] for k in ("epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy", "train_auc", "val_auc", "train_logloss", "val_logloss")}
        best = (np.inf, w.copy(), b, 0); bad = 0
        for ep in range(1, self.n_epochs + 1):
            lr = self.learning_rate / (1.0 + self.lr_decay * (ep - 1))
            order = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = order[s:s + self.batch_size]
                p = _sigmoid(Z[idx] @ w + b)
                dz = cw[idx] * (p - y[idx]) / cw[idx].sum()                # delta rule
                gw = Z[idx].T @ dz + self.l2 * w; gb = dz.sum()
                vw = self.momentum * vw - lr * gw; vb = self.momentum * vb - lr * gb
                w += vw; b += vb
            p_tr = _sigmoid(Z @ w + b)
            tl = _wlogloss(y, p_tr, cw) + 0.5 * self.l2 * float(w @ w)
            hist["epoch"].append(ep); hist["train_loss"].append(tl); hist["train_logloss"].append(_wlogloss(y, p_tr, np.ones(n)))
            hist["train_accuracy"].append(float(((p_tr >= 0.5).astype(int) == y).mean())); hist["train_auc"].append(_auc(y, p_tr))
            if Zv is not None:
                p_v = _sigmoid(Zv @ w + b); vl = _wlogloss(yv, p_v, cwv) + 0.5 * self.l2 * float(w @ w)
                hist["val_loss"].append(vl); hist["val_accuracy"].append(float(((p_v >= 0.5).astype(int) == yv).mean()))
                hist["val_auc"].append(_auc(yv, p_v)); hist["val_logloss"].append(_wlogloss(yv, p_v, np.ones(len(yv))))
                if vl < best[0] - 1e-7:
                    best = (vl, w.copy(), b, ep); bad = 0
                else:
                    bad += 1
                if early_stopping_rounds and bad >= early_stopping_rounds:
                    if verbose:
                        print(f"  early stop at epoch {ep} (best {best[3]})")
                    break
            if verbose and ep % 25 == 0:
                print(f"  epoch {ep:>3} | train loss {tl:.4f}" + (f" | val loss {hist['val_loss'][-1]:.4f}" if Zv is not None else ""))
        if Zv is not None:
            w, b = best[1], best[2]; self.best_epoch_ = best[3]
        else:
            self.best_epoch_ = len(hist["epoch"])
        self.coef_, self.intercept_ = w.copy(), float(b)
        self.history_ = hist; self.n_epochs_run_ = len(hist["epoch"]); self.fit_time_ = time.time() - t0
        self.feature_importances_ = np.abs(self.coef_) / max(np.abs(self.coef_).sum(), 1e-12)
        self.best_threshold_ = 0.5
        return self

    def decision_function(self, X):
        return self._z(X) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    # --- plug-in names used by run_classification.py (unweighted log-loss curves) ---
    @property
    def train_loss(self):
        return self.history_.get("train_logloss") if getattr(self, "history_", None) else None

    @property
    def evals_result_(self):
        h = getattr(self, "history_", None) or {}
        return {"validation": {"logloss": h.get("val_logloss", [])}}

    def get_state(self):
        return {"cls": "SingleLayerPerceptronScratch", "params": dict(learning_rate=self.learning_rate, n_epochs=self.n_epochs, batch_size=self.batch_size,
                                                                       momentum=self.momentum, l2=self.l2, lr_decay=self.lr_decay,
                                                                       scale_pos_weight=self.scale_pos_weight, random_state=self.random_state),
                "mean": self.mean_, "scale": self.scale_, "coef": self.coef_, "intercept": self.intercept_,
                "feature_names": self.feature_names_, "best_threshold": float(self.best_threshold_), "history": self.history_}

    @classmethod
    def from_state(cls, s):
        m = cls(**s["params"]); m.mean_, m.scale_, m.coef_, m.intercept_ = s["mean"], s["scale"], s["coef"], s["intercept"]
        m.feature_names_ = s["feature_names"]; m.best_threshold_ = s["best_threshold"]; m.history_ = s["history"]
        m.feature_importances_ = np.abs(m.coef_) / max(np.abs(m.coef_).sum(), 1e-12)
        return m
