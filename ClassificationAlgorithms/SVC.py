
import os
import sys
import time

import numpy as np


# -----------------------------------------------------------------------------
# small shared helpers (kept local on purpose: this file imports nothing from Common/)
# -----------------------------------------------------------------------------
def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def sqrt_scale_pos_weight(y):
    """sqrt((n - n_pos) / n_pos)  (~2.29 here)."""
    y = np.asarray(y, dtype=int).ravel()
    n_pos = int(y.sum())
    return 1.0 if n_pos in (0, len(y)) else float(np.sqrt((len(y) - n_pos) / n_pos))


def _weights(y, scale_pos_weight, sample_weight):

    y = np.asarray(y).astype(int)
    n_pos = int(np.sum(y == 1))
    n_neg = int(len(y) - n_pos)
    if scale_pos_weight == "auto":
        spw = sqrt_scale_pos_weight(y)
    elif scale_pos_weight == "balanced":
        spw = float(n_neg) / max(1, n_pos)
    else:
        spw = float(scale_pos_weight)
    if n_pos == 0 or n_neg == 0:          # single-class data: nothing to rebalance
        spw = 1.0
    w = np.where(y == 1, spw, 1.0)
    if sample_weight is not None:
        w = w * np.asarray(sample_weight, dtype=float)
    return w, spw


def _auc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]
    i = 0
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


def _kernel_matrix(A, B, kind, gamma, degree, coef0):
    if kind == "linear":
        return A @ B.T
    if kind == "rbf":
        sq = (np.sum(A * A, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :] - 2.0 * (A @ B.T))
        return np.exp(-gamma * np.maximum(sq, 0.0))
    if kind == "poly":
        return (gamma * (A @ B.T) + coef0) ** degree
    raise ValueError('kernel must be "linear", "rbf" or "poly"')


class SVCScratch:
    def __init__(self, C=1.0, kernel="linear", gamma="scale", degree=3, coef0=0.0, solver=None,
                 learning_rate=None, n_iterations=800, tol=1e-4, standardize=True, class_weight=None,
                 scale_pos_weight="balanced", dual_max_samples=7000, dual_max_matrix=3000,
                 store_history=True, random_state=42):
        if kernel not in ("linear", "rbf", "poly"):
            raise ValueError('kernel must be "linear", "rbf" or "poly"')
        if solver is not None and solver not in ("primal", "dual"):
            raise ValueError('solver must be "primal", "dual" or None')
        if solver == "primal" and kernel != "linear":
            raise ValueError('the primal solver only supports the linear kernel (use solver="dual")')
        if isinstance(scale_pos_weight, str) and scale_pos_weight not in ("auto", "balanced"):
            raise ValueError('scale_pos_weight must be "auto", "balanced" or a number')
        self.C = float(C)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.solver = solver or ("primal" if kernel == "linear" else "dual")
        self.learning_rate = learning_rate
        self.n_iterations = int(n_iterations)
        self.tol = float(tol)
        self.standardize = bool(standardize)
        self.class_weight = class_weight
        self.scale_pos_weight = scale_pos_weight
        self.dual_max_samples = int(dual_max_samples)
        self.dual_max_matrix = int(dual_max_matrix)
        self.store_history = bool(store_history)
        self.random_state = random_state

    
    def _fit_scaler(self, X):
        self.mean_ = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        self.scale_ = sd

    def _z(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_

    def _resolve_gamma(self, X):
        if self.gamma == "scale":
            return 1.0 / max(X.shape[1] * X.var(), 1e-12)
        if self.gamma == "auto":
            return 1.0 / max(X.shape[1], 1)
        return float(self.gamma)

    def _fit_platt(self, S, y, cw):

        self._platt_sd = float(S.std()) or 1.0
        s = S / self._platt_sd
        A, B, ridge = 0.0, 0.0, 1e-3
        for _ in range(60):
            p = _sigmoid(A * s + B)
            g = cw * (p - y)
            h = cw * p * (1 - p) + 1e-12
            gr = np.array([np.sum(g * s) + ridge * A, np.sum(g)])
            H = np.array([[np.sum(h * s * s) + ridge, np.sum(h * s)], [np.sum(h * s), np.sum(h)]]) + 1e-9 * np.eye(2)
            step = np.linalg.solve(H, gr)
            A, B = A - step[0], B - step[1]
            if np.abs(step).max() < 1e-9:
                break
        self.platt_ = (float(A), float(B), self._platt_sd)

    # ------------------------------------------------------------------ primal
    def _fit_primal(self, Z, y, cw, Zv=None, yv=None, verbose=False, early_stopping_rounds=None):

        n, d = Z.shape
        t = 2 * y - 1
        self.converged_ = False
        lam = 1.0 / (self.C * n)
        w = np.zeros(d)
        b = 0.0
        lr = self.learning_rate
        if lr is None:                                  
            lr = float(np.clip(1.0 / max(np.mean(cw * np.sum(Z * Z, axis=1)), 1e-9), 1e-3, 1.0))
        hist = {"iter": [], "train_hinge": [], "train_logloss": [], "train_auc": [],
                "val_logloss": [], "val_auc": [], "n_support": []}
        it = 0
        best_val, best_round, patience0 = np.inf, 0, 25
        w_best, b_best = w.copy(), b
        w_sum = np.zeros(d)
        b_sum = 0.0
        n_avg = 0
        t0_avg = max(1, self.n_iterations // 2)
        for it in range(1, self.n_iterations + 1):
            lr_t = lr / (1.0 + it / max(1.0, t0_avg))
            z = Z @ w + b
            margin = 1.0 - t * z
            active = margin > 0
            gw = lam * w - (Z[active].T @ (cw[active] * t[active])) / n
            gb = -float(np.sum(cw[active] * t[active])) / n
            gnorm = float(np.hypot(np.linalg.norm(gw), gb))
            w -= lr_t * gw
            b -= lr_t * gb
            if it >= t0_avg:
                w_sum += w
                b_sum += b
                n_avg += 1

            if Zv is not None and early_stopping_rounds is not None and it % patience0 == 0:
                vl = _wlogloss(yv, _sigmoid(Zv @ w + b), np.ones(len(yv)))
                if vl < best_val - 1e-12:
                    best_val, best_round, w_best, b_best = vl, it, w.copy(), b
                elif (it - best_round) >= early_stopping_rounds * patience0:
                    w, b = w_best, b_best
                    self.early_stopping_used_ = True
                    break

            if self.store_history or it % 10 == 0:
                z = Z @ w + b
                hinge = float(np.mean(cw * np.maximum(0, 1 - t * z)))
                hist["iter"].append(it)
                hist["train_hinge"].append(hinge)
                hist["train_logloss"].append(_wlogloss(y, _sigmoid(z), cw))
                hist["train_auc"].append(_auc(y, z))
                hist["n_support"].append(int(np.sum(t * z < 1.0)))
                if Zv is not None:
                    zv = Zv @ w + b
                    hist["val_logloss"].append(_wlogloss(yv, _sigmoid(zv), np.ones(len(yv))))
                    hist["val_auc"].append(_auc(yv, zv))
            if verbose and it % 50 == 0:
                print(f"  primal iter {it:>4} | hinge {hist['train_hinge'][-1]:.5f} | |grad| {gnorm:.2e}")
            if gnorm < self.tol:
                self.converged_ = True
                break
        else:
            self.converged_ = False
        if n_avg > 0:                                    
            w, b = w_sum / n_avg, b_sum / n_avg
        self.coef_, self.intercept_ = w.copy(), float(b)
        self.n_iterations_ = it
        self.history_ = hist
        z = Z @ w + b
        self.margin_ = float(np.min(t * z))
        self.n_support_ = int(np.sum(t * z < 1.0))
        self.support_ = np.where(t * z < 1.0)[0]
        self.dual_coef_ = None           
        self._train_Z = None
        return self

    # -------------------------------------------------------------------- dual
    def _fit_dual(self, Z, y, cw, Zv=None, yv=None, verbose=False, early_stopping_rounds=None):
        n = len(y)
        t = 2 * y - 1
        self.converged_ = False
        gamma = self._resolve_gamma(Z)
        if self.kernel == "linear" and n <= self.dual_max_matrix:
            self._train_Z = None
            K = Z @ Z.T
            def k_col(j):
                return K[:, j]
            def k_dot(i, j):
                return float(K[i, j])
        elif self.kernel == "linear":
            self._train_Z = None
            def k_col(j):
                return Z @ Z[j]
            def k_dot(i, j):
                return float(Z[i] @ Z[j])
        else:
            if n > self.dual_max_samples:
                raise ValueError(f"the dual solver with kernel='{self.kernel}' needs an n x n kernel "
                                 f"matrix; n={n} > dual_max_samples={self.dual_max_samples}. Subsample, "
                                 f"or use kernel='linear' (implicit, no matrix needed).")
            self._train_Z = Z
            K = _kernel_matrix(Z, Z, self.kernel, gamma, self.degree, self.coef0)
            def k_col(j):
                return K[:, j]
            def k_dot(i, j):
                return float(K[i, j])

        upper = self.C * cw
        a = np.zeros(n)
        b = 0.0
        E = -t.copy()                   
        hist = {"pass": [], "train_hinge": [], "train_logloss": [], "train_auc": [],
                "val_logloss": [], "val_auc": [], "n_support": []}
        rng = np.random.RandomState(self.random_state)
        best_val, best_pass, a_best, b_best = np.inf, 0, a.copy(), b

        for p in range(1, self.n_iterations + 1):
            changed = 0
            for _ in range(n):                      

                H = -E
                up_ok = ((t > 0) & (a < upper - 1e-12)) | ((t < 0) & (a > 1e-12))
                lo_ok = ((t > 0) & (a > 1e-12)) | ((t < 0) & (a < upper - 1e-12))
                if not (up_ok.any() and lo_ok.any()):
                    break
                up_idx = np.flatnonzero(up_ok)
                lo_idx = np.flatnonzero(lo_ok)
                i = int(up_idx[np.argmax(H[up_ok])])
                j = int(lo_idx[np.argmin(H[lo_ok])])
                if H[i] - H[j] <= self.tol:          
                    break
            
                cands = [(i, j)]
                for ii in up_idx[np.argsort(-H[up_ok])[:4]]:
                    for jj in lo_idx[np.argsort(H[lo_ok])[:4]]:
                        if (int(ii), int(jj)) not in cands:
                            cands.append((int(ii), int(jj)))
                best = None
                for (ii, jj) in cands:
                    if ii == jj:
                        continue
                    eta = 2.0 * k_dot(ii, jj) - k_dot(ii, ii) - k_dot(jj, jj)
                    if eta >= 0:
                        continue
                    d_star = (E[ii] - E[jj]) / eta
                    lo, hi = -np.inf, np.inf
                    ok = True
                    for k, sgn in ((ii, 1.0), (jj, -1.0)):
                        c = t[k] * sgn
                        if c > 0:
                            lo = max(lo, (0.0 - a[k]) / c)
                            hi = min(hi, (upper[k] - a[k]) / c)
                        elif c < 0:
                            lo = max(lo, (upper[k] - a[k]) / c)
                            hi = min(hi, (0.0 - a[k]) / c)
                        else:
                            ok = False
                    if not ok or lo >= hi:
                        continue
                    delta = float(np.clip(d_star, lo, hi))
                    if abs(delta) < 1e-12:
                        continue
                    gain = (E[jj] - E[ii]) * delta + 0.5 * eta * delta * delta
                    if gain <= 0:
                        continue
                    if best is None or gain > best[0]:
                        best = (gain, ii, jj, delta)
                if best is None:
                    break
                _, i, j, delta = best
                d_ai, d_aj = t[i] * delta, -t[j] * delta
                ai_old, aj_old, b_old = a[i], a[j], b
                a[i] = ai_old + d_ai
                a[j] = aj_old + d_aj
                b1 = b - E[i] - t[i] * d_ai * k_dot(i, i) - t[j] * d_aj * k_dot(i, j)
                b2 = b - E[j] - t[i] * d_ai * k_dot(i, j) - t[j] * d_aj * k_dot(j, j)
                if 0 < a[i] < upper[i]:
                    b = b1
                elif 0 < a[j] < upper[j]:
                    b = b2
                else:
                    b = 0.5 * (b1 + b2)
                E = E + (d_ai * t[i]) * k_col(i) + (d_aj * t[j]) * k_col(j) + (b - b_old)
                changed += 1

            at = a * t
            if self.kernel == "linear":
                w_now = Z.T @ at
                z = Z @ w_now + b
            else:
                z = K @ at + b
            hinge = float(np.mean(cw * np.maximum(0, 1 - t * z)))
            hist["pass"].append(p)
            hist["train_hinge"].append(hinge)
            hist["train_logloss"].append(_wlogloss(y, _sigmoid(z), cw))
            hist["train_auc"].append(_auc(y, z))
            hist["n_support"].append(int(np.sum(a > 1e-8)))
            if Zv is not None:
                zv = (Zv @ w_now + b) if self.kernel == "linear" else self._scores_alpha(Zv, Z, at, b, gamma)
                vl = _wlogloss(yv, _sigmoid(zv), np.ones(len(yv)))
                hist["val_logloss"].append(vl)
                hist["val_auc"].append(_auc(yv, zv))
                if vl < best_val - 1e-12:
                    best_val, best_pass, a_best, b_best = vl, p, a.copy(), b
                elif early_stopping_rounds is not None and (p - best_pass) >= early_stopping_rounds:
                    a, b = a_best, b_best
                    self.early_stopping_used_ = True
                    break
            up_ok = ((t > 0) & (a < upper - 1e-12)) | ((t < 0) & (a > 1e-12))
            lo_ok = ((t > 0) & (a > 1e-12)) | ((t < 0) & (a < upper - 1e-12))
            gap = np.inf
            if up_ok.any() and lo_ok.any():
                Hv = -E
                gap = float(Hv[up_ok].max() - Hv[lo_ok].min())
            if verbose:
                print(f"  smo pass {p:>3} | updates {changed:>5} | hinge {hinge:.5f} "
                      f"| support vectors {hist['n_support'][-1]} | KKT gap {gap:.2e}")
            if changed == 0 or gap <= self.tol:
                self.converged_ = True
                break

        at = a * t
        self.support_ = np.where(a > 1e-8)[0]
        self.dual_coef_ = (a[self.support_] * t[self.support_]).copy()
        z_train = (Z @ (Z.T @ at)) if self.kernel == "linear" else K @ at
        r_all = t - z_train                      # r_i = t_i - w.z_i  (b = r_i makes y_i f_i = 1)
        free = (a > 1e-12) & (a < upper - 1e-12)
        if free.any():
            b = float(np.mean(r_all[free]))
        else:
            at0 = a <= 1e-12
            atC = a >= upper - 1e-12
            lo_parts = [r_all[at0 & (t > 0)], r_all[atC & (t < 0)]]
            hi_parts = [r_all[at0 & (t < 0)], r_all[atC & (t > 0)]]
            lo_parts = [p for p in lo_parts if p.size]
            hi_parts = [p for p in hi_parts if p.size]
            lo_b = float(max(p.max() for p in lo_parts)) if lo_parts else -np.inf
            hi_b = float(min(p.min() for p in hi_parts)) if hi_parts else np.inf
            if np.isfinite(lo_b) and np.isfinite(hi_b) and lo_b <= hi_b:
                b = 0.5 * (lo_b + hi_b)
        self.intercept_ = float(b)
        self.n_support_ = len(self.support_)
        self.n_iterations_ = len(hist["pass"])
        self._gamma_used = gamma
        self.margin_ = float(np.min(t * ((Z @ (Z.T @ at) + b) if self.kernel == "linear" else K @ at + b)))
        self.coef_ = (Z.T @ at).copy() if self.kernel == "linear" else None
        self.history_ = hist
        return self

    def _scores_alpha(self, Znew, Z, at, b, gamma):
        
        return _kernel_matrix(Znew, Z, self.kernel, gamma, self.degree, self.coef0) @ at + b

    # --------------------------------------------------------------------- fit
    def fit(self, X, y, eval_set=None, feature_names=None, early_stopping_rounds=30,
            sample_weight=None, verbose=False, **_ignored):

        t0 = time.time()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int).ravel()
        self._fit_scaler(X)
        Z = self._z(X)
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(Z.shape[1])]
        cw, self.scale_pos_weight_ = _weights(y, self.scale_pos_weight, sample_weight)
        if self.class_weight == "balanced":
            n0, n1 = int((y == 0).sum()), int((y == 1).sum())
            if n0 and n1:
                cw = cw * np.where(y == 1, len(y) / (2.0 * n1), len(y) / (2.0 * n0))
        elif self.class_weight not in (None,):
            raise ValueError('class_weight must be None or "balanced"')

        Zv = yv = None
        if eval_set:
            Zv = self._z(eval_set[0][0])
            yv = np.asarray(eval_set[0][1]).astype(int).ravel()

        if self.solver == "primal":
            self._fit_primal(Z, y, cw, Zv=Zv, yv=yv, verbose=verbose,
                             early_stopping_rounds=early_stopping_rounds)
        else:
            self._fit_dual(Z, y, cw, Zv=Zv, yv=yv, verbose=verbose,
                           early_stopping_rounds=early_stopping_rounds)


        if eval_set:
            S = self.decision_function(np.asarray(eval_set[0][0], dtype=float))
            self._fit_platt(S, yv, np.ones(len(S)))
        else:
            S = self.decision_function(X)
            self._fit_platt(S, y, np.ones(len(S)))
        self.best_threshold_ = 0.5

      
        if self.kernel == "linear":
            imp = np.abs(self.coef_)
        else:
            imp = self._sensitivity(Z)
        self.feature_importances_ = imp / max(imp.sum(), 1e-12)
        self.fit_time_ = time.time() - t0
        return self

    # ---------------------------------------------------------------- predict
    def decision_function(self, X):
       
        Z = self._z(X)
        if self.kernel != "linear":
          
            sv = self._train_Z[self.support_]
            return (_kernel_matrix(Z, sv, self.kernel, self._gamma_used, self.degree, self.coef0)
                    @ self.dual_coef_ + self.intercept_)
        return Z @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p = _sigmoid(self.platt_[0] * self.decision_function(X) / self.platt_[2] + self.platt_[1])
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=None):
        t = self.best_threshold_ if threshold is None else threshold
        return (self.predict_proba(X)[:, 1] >= t).astype(int)

    def predict_margin(self, X):
       
        return (self.decision_function(X) > 0).astype(int)

    # ------------------------------------------------------------- report bits
    def _sensitivity(self, Z, h=0.25, max_rows=None):
       
        max_rows = max_rows or (1200 if self.kernel == "linear" else 400)
        Zs = Z[:max_rows]
        base = self.decision_function(self._unz(Zs))
        imp = np.zeros(Z.shape[1])
        for j in range(Z.shape[1]):
            Zp = Zs.copy()
            Zp[:, j] += h
            imp[j] = np.mean(np.abs(self.decision_function(self._unz(Zp)) - base)) / h
        return imp

    def _unz(self, Z):
        return Z * self.scale_ + self.mean_

    def coefficient_table(self, top=None):
        
        if self.coef_ is None:
            raise ValueError("coefficient_table is only defined for the linear kernel; use "
                             "feature_importance_table() for rbf/poly")
        order = np.argsort(-np.abs(self.coef_))
        rows = [{"feature": self.feature_names_[j], "weight_per_1SD": float(self.coef_[j])} for j in order]
        return rows[:top] if top else rows

    def feature_importance_table(self, top=None):
        rows = [(self.feature_names_[j], float(self.feature_importances_[j]))
                for j in np.argsort(-self.feature_importances_)]
        return rows[:top] if top else rows

    def explain(self, x, top=5):

        x = np.asarray(x, dtype=float).reshape(1, -1)
        z = self._z(x)[0]
        out = {"score": float(self.decision_function(x)[0]),
               "p_default": float(self.predict_proba(x)[0, 1])}
        if self.kernel == "linear":
            contrib = self.coef_ * z
            order = np.argsort(-np.abs(contrib))[:top]
            out["contributions"] = [(self.feature_names_[j], float(contrib[j])) for j in order]
        else:
            out["nearest_support_vectors"] = [int(i) for i in self.support_[:top]]
        return out

    def pick_threshold(self, X_val, y_val, metric="f1"):
        
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

    # ------------------------------------------------- plug-in names for the runner
    @property
    def train_loss(self):
        return self.history_.get("train_logloss") if getattr(self, "history_", None) else None

    @property
    def evals_result_(self):
        h = getattr(self, "history_", None) or {}
        return {"validation": {"logloss": h.get("val_logloss", []), "auc": h.get("val_auc", [])}}

    # ----------------------------------------------------------- persistence
    def get_state(self):
        return {"cls": "SVCScratch",
                "params": dict(C=self.C, kernel=self.kernel, gamma=self.gamma, degree=self.degree,
                               coef0=self.coef0, solver=self.solver, learning_rate=self.learning_rate,
                               n_iterations=self.n_iterations, tol=self.tol, standardize=self.standardize,
                               class_weight=self.class_weight, scale_pos_weight=self.scale_pos_weight,
                               dual_max_samples=self.dual_max_samples, store_history=self.store_history,
                               random_state=self.random_state),
                "mean": self.mean_, "scale": self.scale_, "coef": self.coef_,
                "intercept": self.intercept_, "platt": list(self.platt_),
                "feature_names": self.feature_names_, "importances": self.feature_importances_,
                "best_threshold": float(self.best_threshold_), "history": self.history_,
                "support": self.support_, "dual_coef": self.dual_coef_,
                "gamma_used": getattr(self, "_gamma_used", None),
                "train_Z": getattr(self, "_train_Z", None)}

    @classmethod
    def from_state(cls, s):
        m = cls(**s["params"])
        m.mean_, m.scale_, m.coef_, m.intercept_ = s["mean"], s["scale"], s["coef"], s["intercept"]
        m.platt_ = tuple(s["platt"])
        m.feature_names_, m.feature_importances_ = s["feature_names"], s["importances"]
        m.best_threshold_, m.history_ = s["best_threshold"], s["history"]
        m.support_, m.dual_coef_ = s["support"], s["dual_coef"]
        m._gamma_used = s["gamma_used"]
        m._train_Z = s["train_Z"]
        m.n_support_ = 0 if m.support_ is None else len(m.support_)
        return m

    def save(self, path):
        import joblib
        joblib.dump({"model": "SVCScratch", "state": self.get_state()}, path)

    @classmethod
    def load(cls, path):
        import joblib
        return cls.from_state(joblib.load(path)["state"])


# convenience alias (the runner / reports may use either name)
SupportVectorMachineScratch = SVCScratch
