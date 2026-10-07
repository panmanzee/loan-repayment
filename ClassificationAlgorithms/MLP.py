"""
Multi-Layer Perceptron (from scratch) - Classification task (target: not.fully.paid, 1 = defaulted)

Everything is written from scratch with NumPy (forward pass, back-propagation, Adam). The file is self-contained
(it does not import Common/).

THE MODEL
---------
A stack of fully-connected layers. With hidden sizes (32, 16) and 23 inputs:

        a0 = x                                        (standardised features, 23)
        z1 = a0 W1 + b1      a1 = relu(z1)            (32 hidden neurons)
        z2 = a1 W2 + b2      a2 = relu(z2)            (16 hidden neurons)
        z3 = a2 W3 + b3      p  = sigmoid(z3)         (1 output neuron = P(default))

Each hidden neuron is the SLP of Perceptron.py followed by a non-linearity; stacking them lets the network bend the
decision boundary (interactions such as "high utilisation AND many inquiries"), which a single neuron cannot do.

LOSS (what is minimised)
------------------------
Weighted binary cross-entropy + L2 penalty:

        L = -(1/W) sum_i c_i [ y_i log p_i + (1-y_i) log(1-p_i) ]  +  (lambda/2) sum_layers ||W||^2        c_i = sample weight, W = sum c_i

BACK-PROPAGATION (chain rule, layer by layer, from the output back to the input)
--------------------------------------------------------------------------------
        output layer :   delta_L = c_i (p_i - y_i) / W                         <- sigmoid + cross-entropy collapse to this
        hidden layer :   delta_l = (delta_{l+1} W_{l+1}^T)  *  f'(z_l)          f'(z) = 1[z>0] for ReLU, 1 - tanh^2 for tanh
        gradients    :   dW_l = a_{l-1}^T delta_l + lambda W_l ,   db_l = sum_i delta_l
`gradient_check()` compares these analytic gradients with finite differences (the standard correctness test).

OPTIMISER: Adam (per-weight adaptive step: m = b1 m + (1-b1) g ; v = b2 v + (1-b2) g^2 ; W -= lr * m_hat / (sqrt(v_hat) + eps));
"sgd" and "momentum" are available too. Weights start with He initialisation (std = sqrt(2 / fan_in)) for ReLU.
Optional inverted dropout on the hidden layers, mini-batches, early stopping on the validation loss with automatic
restoring of the best epoch.

Class imbalance: `scale_pos_weight="auto"` weights defaulters sqrt((n-n_pos)/n_pos) times more (the project convention).

Example
-------
    mlp = MLPClassifierScratch(hidden_layers=(32, 16), learning_rate=1e-3, l2=1e-3, n_epochs=200)
    mlp.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], feature_names=names, early_stopping_rounds=20)
    proba = mlp.predict_proba(X_te)[:, 1]
    mlp.history_["val_loss"]            # loss curve
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


_ACT = {"relu": (lambda z: np.maximum(z, 0.0), lambda z, a: (z > 0).astype(float)),
        "tanh": (np.tanh, lambda z, a: 1.0 - a * a),
        "sigmoid": (_sigmoid, lambda z, a: a * (1.0 - a))}


class MLPClassifierScratch:
    def __init__(self, hidden_layers=(32, 16), activation="relu", learning_rate=1e-3, n_epochs=200, batch_size=128,
                 l2=1e-3, dropout=0.0, optimizer="adam", beta1=0.9, beta2=0.999, momentum=0.9,
                 scale_pos_weight="auto", random_state=42):
        if activation not in _ACT:
            raise ValueError("activation must be relu, tanh or sigmoid")
        if optimizer not in ("adam", "sgd", "momentum"):
            raise ValueError("optimizer must be adam, sgd or momentum")
        self.hidden_layers = tuple(hidden_layers); self.activation = activation; self.learning_rate = learning_rate
        self.n_epochs = n_epochs; self.batch_size = batch_size; self.l2 = l2; self.dropout = dropout
        self.optimizer = optimizer; self.beta1 = beta1; self.beta2 = beta2; self.momentum = momentum
        self.scale_pos_weight = scale_pos_weight; self.random_state = random_state

    # ---------------------------------------------------------- init / helpers
    def _fit_scaler(self, X):
        self.mean_ = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1.0; self.scale_ = sd

    def _z(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_

    def _init_params(self, d, rng):
        sizes = [d, *self.hidden_layers, 1]
        self.W_ = [rng.normal(0.0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.b_ = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    def _forward(self, Z, train=False, rng=None):
        f = _ACT[self.activation][0]
        acts, zs, masks = [Z], [], []
        a = Z
        for l in range(len(self.W_) - 1):
            z = a @ self.W_[l] + self.b_[l]; a = f(z)
            if train and self.dropout > 0:
                m = (rng.rand(*a.shape) >= self.dropout) / (1.0 - self.dropout); a = a * m; masks.append(m)
            else:
                masks.append(None)
            zs.append(z); acts.append(a)
        z = a @ self.W_[-1] + self.b_[-1]; zs.append(z)
        return _sigmoid(z).ravel(), acts, zs, masks

    def _backward(self, y, w_norm, p, acts, zs, masks):
        """Returns gradients [dW...], [db...] for the (already normalised) weighted cross-entropy + L2."""
        fprime = _ACT[self.activation][1]
        delta = (w_norm * (p - y))[:, None]                      # sigmoid + cross-entropy
        gW, gb = [None] * len(self.W_), [None] * len(self.W_)
        for l in range(len(self.W_) - 1, -1, -1):
            gW[l] = acts[l].T @ delta + self.l2 * self.W_[l]; gb[l] = delta.sum(axis=0)
            if l > 0:
                delta = delta @ self.W_[l].T
                if masks[l - 1] is not None:
                    delta = delta * masks[l - 1]
                delta = delta * fprime(zs[l - 1], acts[l])
        return gW, gb

    def _loss(self, y, p, w):
        return _wlogloss(y, p, w) + 0.5 * self.l2 * sum(float((W * W).sum()) for W in self.W_)

    # --------------------------------------------------------------------- fit
    def fit(self, X, y, eval_set=None, feature_names=None, early_stopping_rounds=20, sample_weight=None, verbose=False, **_ignored):
        t0 = time.time()
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y).astype(float).ravel()
        self._fit_scaler(X); Z = self._z(X); n, d = Z.shape
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(d)]
        spw = sqrt_scale_pos_weight(y) if self.scale_pos_weight == "auto" else float(self.scale_pos_weight)
        self.scale_pos_weight_ = spw
        cw = np.where(y == 1, spw, 1.0)
        if sample_weight is not None:
            cw = cw * np.asarray(sample_weight, dtype=float)
        rng = np.random.RandomState(self.random_state)
        self._init_params(d, rng)
        L = len(self.W_)
        mW = [np.zeros_like(W) for W in self.W_]; mb = [np.zeros_like(b) for b in self.b_]
        vW = [np.zeros_like(W) for W in self.W_]; vb = [np.zeros_like(b) for b in self.b_]
        Zv = yv = cwv = None
        if eval_set:
            Zv, yv = self._z(eval_set[0][0]), np.asarray(eval_set[0][1]).astype(float)
            cwv = np.where(yv == 1, spw, 1.0)
        hist = {k: [] for k in ("epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy", "train_auc", "val_auc", "train_logloss", "val_logloss")}
        best = (np.inf, None, 0); bad = 0; step = 0
        for ep in range(1, self.n_epochs + 1):
            order = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = order[s:s + self.batch_size]; step += 1
                p, acts, zs, masks = self._forward(Z[idx], train=True, rng=rng)
                gW, gb = self._backward(y[idx], cw[idx] / cw[idx].sum(), p, acts, zs, masks)
                for l in range(L):
                    if self.optimizer == "adam":
                        mW[l] = self.beta1 * mW[l] + (1 - self.beta1) * gW[l]; vW[l] = self.beta2 * vW[l] + (1 - self.beta2) * gW[l] ** 2
                        mb[l] = self.beta1 * mb[l] + (1 - self.beta1) * gb[l]; vb[l] = self.beta2 * vb[l] + (1 - self.beta2) * gb[l] ** 2
                        c1, c2 = 1 - self.beta1 ** step, 1 - self.beta2 ** step
                        self.W_[l] -= self.learning_rate * (mW[l] / c1) / (np.sqrt(vW[l] / c2) + 1e-8)
                        self.b_[l] -= self.learning_rate * (mb[l] / c1) / (np.sqrt(vb[l] / c2) + 1e-8)
                    elif self.optimizer == "momentum":
                        mW[l] = self.momentum * mW[l] - self.learning_rate * gW[l]; mb[l] = self.momentum * mb[l] - self.learning_rate * gb[l]
                        self.W_[l] += mW[l]; self.b_[l] += mb[l]
                    else:
                        self.W_[l] -= self.learning_rate * gW[l]; self.b_[l] -= self.learning_rate * gb[l]
            p_tr = self._forward(Z)[0]
            hist["epoch"].append(ep); hist["train_loss"].append(self._loss(y, p_tr, cw)); hist["train_logloss"].append(_wlogloss(y, p_tr, np.ones(n)))
            hist["train_accuracy"].append(float(((p_tr >= 0.5) == (y == 1)).mean())); hist["train_auc"].append(_auc(y, p_tr))
            if Zv is not None:
                p_v = self._forward(Zv)[0]; vl = self._loss(yv, p_v, cwv)
                hist["val_loss"].append(vl); hist["val_accuracy"].append(float(((p_v >= 0.5) == (yv == 1)).mean())); hist["val_auc"].append(_auc(yv, p_v))
                hist["val_logloss"].append(_wlogloss(yv, p_v, np.ones(len(yv))))
                if vl < best[0] - 1e-7:
                    best = (vl, ([W.copy() for W in self.W_], [b.copy() for b in self.b_]), ep); bad = 0
                else:
                    bad += 1
                if early_stopping_rounds and bad >= early_stopping_rounds:
                    if verbose:
                        print(f"  early stop at epoch {ep} (best epoch {best[2]})")
                    break
            if verbose and ep % 20 == 0:
                print(f"  epoch {ep:>3} | train loss {hist['train_loss'][-1]:.4f}" + (f" | val loss {hist['val_loss'][-1]:.4f}" if Zv is not None else ""))
        if Zv is not None and best[1] is not None:
            self.W_, self.b_ = best[1]; self.best_epoch_ = best[2]
        else:
            self.best_epoch_ = len(hist["epoch"])
        self.history_ = hist; self.n_epochs_run_ = len(hist["epoch"]); self.fit_time_ = time.time() - t0
        self.n_parameters_ = int(sum(W.size + b.size for W, b in zip(self.W_, self.b_)))
        # "importance": mean |dp/dx_j| (average sensitivity of the output to each standardised input), by finite difference
        self.feature_importances_ = self._sensitivity(Z)
        self.best_threshold_ = 0.5
        return self

    def _sensitivity(self, Z, h=0.25, max_rows=1500):
        Zs = Z[:max_rows]; base = self._forward(Zs)[0]; imp = np.zeros(Z.shape[1])
        for j in range(Z.shape[1]):
            Zp = Zs.copy(); Zp[:, j] += h
            imp[j] = np.mean(np.abs(self._forward(Zp)[0] - base)) / h
        return imp / max(imp.sum(), 1e-12)


    # --- plug-in names used by run_classification.py (unweighted log-loss curves) ---
    @property
    def train_loss(self):
        return self.history_.get("train_logloss") if getattr(self, "history_", None) else None

    @property
    def evals_result_(self):
        h = getattr(self, "history_", None) or {}
        return {"validation": {"logloss": h.get("val_logloss", [])}}

    # -------------------------------------------------------------- predict
    def predict_proba(self, X):
        p = self._forward(self._z(X))[0]
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

    def feature_importance_table(self, top=None):
        order = np.argsort(-self.feature_importances_)
        rows = [(self.feature_names_[j], float(self.feature_importances_[j])) for j in order]
        return rows[:top] if top else rows

    # --------------------------------------------------------- correctness test
    def gradient_check(self, X, y, n_checks=8, eps=1e-5, seed=0):
        """
        Compare analytic back-prop gradients with central finite differences on a few random weights.
        Returns the largest relative error (should be ~1e-7 or smaller). Dropout is switched off for the check.
        """
        X = np.asarray(X, dtype=np.float64)[:200]; y = np.asarray(y).astype(float)[:200]
        if not hasattr(self, "W_"):
            self._fit_scaler(X); self._init_params(X.shape[1], np.random.RandomState(seed))
        Z = self._z(X); w = np.where(y == 1, 2.0, 1.0); wn = w / w.sum()
        p, acts, zs, masks = self._forward(Z)
        gW, gb = self._backward(y, wn, p, acts, zs, masks)
        rng = np.random.RandomState(seed); worst = 0.0
        for _ in range(n_checks):
            l = rng.randint(len(self.W_)); i = rng.randint(self.W_[l].shape[0]); j = rng.randint(self.W_[l].shape[1])
            old = self.W_[l][i, j]
            self.W_[l][i, j] = old + eps; lp = self._loss(y, self._forward(Z)[0], w)
            self.W_[l][i, j] = old - eps; lm = self._loss(y, self._forward(Z)[0], w)
            self.W_[l][i, j] = old
            num = (lp - lm) / (2 * eps); ana = gW[l][i, j]
            worst = max(worst, abs(num - ana) / max(abs(num) + abs(ana), 1e-12))
        return worst

    # ------------------------------------------------------------ save / load
    def get_state(self):
        return {"cls": "MLPClassifierScratch", "params": dict(hidden_layers=list(self.hidden_layers), activation=self.activation,
                                                               learning_rate=self.learning_rate, n_epochs=self.n_epochs, batch_size=self.batch_size,
                                                               l2=self.l2, dropout=self.dropout, optimizer=self.optimizer, beta1=self.beta1,
                                                               beta2=self.beta2, momentum=self.momentum, scale_pos_weight=self.scale_pos_weight,
                                                               random_state=self.random_state),
                "mean": self.mean_, "scale": self.scale_, "W": self.W_, "b": self.b_, "feature_names": self.feature_names_,
                "importances": self.feature_importances_, "best_threshold": float(self.best_threshold_), "history": self.history_}

    @classmethod
    def from_state(cls, s):
        m = cls(**s["params"]); m.mean_, m.scale_, m.W_, m.b_ = s["mean"], s["scale"], s["W"], s["b"]
        m.feature_names_ = s["feature_names"]; m.feature_importances_ = s["importances"]
        m.best_threshold_ = s["best_threshold"]; m.history_ = s["history"]
        return m

    def save(self, path):
        import joblib
        joblib.dump({"model": "MLPClassifierScratch", "state": self.get_state()}, path)

    @classmethod
    def load(cls, path):
        import joblib
        return cls.from_state(joblib.load(path)["state"])
