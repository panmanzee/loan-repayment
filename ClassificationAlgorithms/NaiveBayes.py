"""
Naive Bayes (from scratch) - Classification task (target: not.fully.paid, 1 = defaulted)

Everything is written from scratch with NumPy. The file is self-contained (it does not import Common/).

THE MODEL
---------
Bayes' theorem turns "how likely is this borrower's profile if they default?" into "how likely is a default
given this profile?":

        P(y | x)  =  P(x | y) * P(y) / P(x)             <- posterior = likelihood * prior / evidence

Writing P(x | y) for 23 features at once would need an enormous table, so Naive Bayes makes ONE assumption:

        P(x | y)  =  P(x_1 | y) * P(x_2 | y) * ... * P(x_d | y)       <- features independent GIVEN the class

so only d small one-dimensional distributions per class have to be estimated. We work in log space so that
23 tiny probabilities do not underflow to 0:

        log P(y=c | x)  ~  log P(y=c)  +  sum_j  log P(x_j | y=c)         (P(x) is the same for both classes)

        P(default | x)  =  sigmoid( score_1 - score_0 )     <- the "evidence" cancels, only a difference is needed

Each feature gets the density that matches its type (kind="mixed", the default):

    * binary feature (0/1, e.g. the one-hot `purpose_*`, `high_revol_util`)   -> Bernoulli with Laplace smoothing
            P(x_j=1 | y=c) = (count_c(x_j=1) + alpha) / (n_c + 2*alpha)
    * numeric feature                                                         -> Gaussian
            P(x_j | y=c) = N(x_j ; mu_jc , sigma_jc^2)      mu, sigma^2 = (weighted) mean / variance inside class c

kind="gaussian"    : every feature Gaussian (this is what sklearn's GaussianNB does).
kind="categorical" : every feature is cut into `n_bins` equal-frequency bins and treated as a discrete variable
                     with Laplace smoothing - the counting version taught in Discrete Mathematics.

WHY IT IS INTERESTING ON THIS DATASET
-------------------------------------
The "naive" assumption is clearly false here (fico, int.rate and fico_rate_gap move together, installment and
income are linked ...), so the probabilities come out over-confident (they are pushed towards 0 / 1) -
but the RANKING of borrowers is still decent, which is what ROC-AUC measures. `explain()` shows each feature's
log-likelihood-ratio contribution, so the prediction is fully traceable.

Class imbalance: `prior="empirical"` uses the observed 84/16 split; `prior="balanced"` uses 50/50;
`prior="sqrt"` (project convention, like the other models) multiplies the positive prior by
sqrt((n-n_pos)/n_pos). The threshold is picked on the validation set anyway (`pick_threshold`).

Example
-------
    nb = NaiveBayesScratch(kind="mixed")
    nb.fit(X_tr, y_tr, feature_names=names)
    proba = nb.predict_proba(X_te)[:, 1]
    print(nb.export_text(top=5))          # strongest evidence features
    nb.save("saved_models/naive_bayes.npz"); nb2 = NaiveBayesScratch.load("saved_models/naive_bayes.npz")
"""
import time

import numpy as np

_LOG_2PI = np.log(2.0 * np.pi)


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))          # numerically stable


class NaiveBayesScratch:
    def __init__(self, kind="mixed", var_smoothing=1e-9, alpha=1.0, n_bins=10, prior="empirical"):
        if kind not in ("mixed", "gaussian", "categorical"):
            raise ValueError("kind must be 'mixed', 'gaussian' or 'categorical'")
        if not (isinstance(prior, (list, tuple, np.ndarray)) or prior in ("empirical", "balanced", "sqrt")):
            raise ValueError("prior must be 'empirical', 'balanced', 'sqrt' or [P(y=0), P(y=1)]")
        self.kind, self.var_smoothing, self.alpha, self.n_bins, self.prior = kind, var_smoothing, alpha, n_bins, prior

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, sample_weight=None, feature_names=None, **_ignored):
        t0 = time.time()
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y).astype(int).ravel()
        n, d = X.shape
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        self.n_features_ = d
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(d)]
        self.classes_ = np.array([0, 1])
        Wc = np.array([w[y == c].sum() for c in (0, 1)])
        emp = Wc / Wc.sum()
        if self.prior == "empirical":
            pri = emp
        elif self.prior == "balanced":
            pri = np.array([0.5, 0.5])
        elif self.prior == "sqrt":
            pri = emp * np.array([1.0, np.sqrt(emp[0] / emp[1])]); pri = pri / pri.sum()
        else:
            pri = np.asarray(self.prior, dtype=float); pri = pri / pri.sum()
        self.class_prior_ = pri
        self.empirical_prior_ = emp

        is_bin = np.array([np.isin(np.unique(X[:, j]), (0.0, 1.0)).all() for j in range(d)])
        if self.kind == "gaussian":
            is_bin[:] = False
        self.feature_type_ = np.where(is_bin, "bernoulli", "gaussian")
        if self.kind == "categorical":
            self.feature_type_ = np.array(["categorical"] * d)

        # Gaussian parameters (all columns; only used by the gaussian ones)
        self.theta_ = np.zeros((2, d)); self.var_ = np.ones((2, d))
        # Bernoulli: P(x=1 | c)
        self.p1_ = np.full((2, d), 0.5)
        # categorical: bin edges per feature and log P(bin | c)
        self.edges_ = [None] * d; self.cat_logp_ = [None] * d

        for c in (0, 1):
            m = y == c; wc = w[m]; sw = wc.sum()
            Xc = X[m]
            mu = (wc[:, None] * Xc).sum(0) / sw
            self.theta_[c] = mu
            self.var_[c] = (wc[:, None] * (Xc - mu) ** 2).sum(0) / sw
        # same variance floor as the textbook / sklearn recipe: epsilon = var_smoothing * (largest feature variance)
        self.epsilon_ = self.var_smoothing * X.var(axis=0).max()
        self.var_ = self.var_ + self.epsilon_

        for j in range(d):
            if self.feature_type_[j] == "bernoulli":
                for c in (0, 1):
                    m = y == c
                    self.p1_[c, j] = ((w[m] * X[m, j]).sum() + self.alpha) / (w[m].sum() + 2 * self.alpha)
            elif self.feature_type_[j] == "categorical":
                qs = np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.n_bins + 1)[1:-1]))
                self.edges_[j] = qs
                b = np.searchsorted(qs, X[:, j], side="right"); K = len(qs) + 1
                lp = np.zeros((2, K))
                for c in (0, 1):
                    m = y == c
                    cnt = np.bincount(b[m], weights=w[m], minlength=K) + self.alpha
                    lp[c] = np.log(cnt / cnt.sum())
                self.cat_logp_[j] = lp

        # per-feature "importance" = symmetric KL divergence between the two class-conditional distributions
        self.feature_importances_ = self._divergence(X)
        self.fit_time_ = time.time() - t0
        self.best_threshold_ = 0.5
        return self

    # --------------------------------------------------------- log-likelihoods
    def _feature_loglik(self, X):
        """Array (n, 2, d): log P(x_j | y=c) for every row, class and feature."""
        X = np.asarray(X, dtype=np.float64); n, d = X.shape
        out = np.zeros((n, 2, d))
        for j in range(d):
            t = self.feature_type_[j]
            for c in (0, 1):
                if t == "gaussian":
                    out[:, c, j] = -0.5 * (_LOG_2PI + np.log(self.var_[c, j])) - 0.5 * (X[:, j] - self.theta_[c, j]) ** 2 / self.var_[c, j]
                elif t == "bernoulli":
                    p = self.p1_[c, j]
                    out[:, c, j] = np.where(X[:, j] >= 0.5, np.log(p), np.log1p(-p))
                else:
                    b = np.searchsorted(self.edges_[j], X[:, j], side="right")
                    out[:, c, j] = self.cat_logp_[j][c][b]
        return out

    def _divergence(self, X):
        ll = self._feature_loglik(X)                      # (n, 2, d)
        # E_x~train[ |log P(x|1) - log P(x|0)| ] : how strongly the feature separates the classes on average
        return np.abs(ll[:, 1, :] - ll[:, 0, :]).mean(axis=0)

    def joint_log_likelihood(self, X):
        """(n, 2) matrix  log P(y=c) + sum_j log P(x_j | y=c)   (the numerator of Bayes' theorem, in logs)."""
        ll = self._feature_loglik(X).sum(axis=2)
        return ll + np.log(self.class_prior_)[None, :]

    def predict_log_proba(self, X):
        jll = self.joint_log_likelihood(X)
        m = jll.max(axis=1, keepdims=True)
        return jll - (m + np.log(np.exp(jll - m).sum(axis=1, keepdims=True)))      # log-sum-exp normalisation

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def decision_function(self, X):
        """log-odds of default = log P(y=1|x) - log P(y=0|x)."""
        jll = self.joint_log_likelihood(X)
        return jll[:, 1] - jll[:, 0]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    # -------------------------------------------------------------- insight
    def explain(self, x, top=5):
        """Why does the model score this one row the way it does? Returns [(feature, log-likelihood-ratio)] sorted."""
        ll = self._feature_loglik(np.atleast_2d(x))[0]
        llr = ll[1] - ll[0]                                # >0 pushes towards default
        order = np.argsort(-np.abs(llr))[:top]
        return [(self.feature_names_[j], float(llr[j])) for j in order]

    def export_text(self, top=5):
        prior = np.log(self.class_prior_[1] / self.class_prior_[0])
        lines = [f"log-odds(default) = {prior:+.3f}  (log prior odds)  +  sum of per-feature evidence", ""]
        order = np.argsort(-self.feature_importances_)[:top]
        for j in order:
            t = self.feature_type_[j]
            if t == "gaussian":
                lines.append(f"{self.feature_names_[j]:<24} Gaussian   mean(paid)={self.theta_[0, j]:.3f}  mean(default)={self.theta_[1, j]:.3f}"
                             f"   evidence={self.feature_importances_[j]:.3f}")
            elif t == "bernoulli":
                lines.append(f"{self.feature_names_[j]:<24} Bernoulli  P(1|paid)={self.p1_[0, j]:.3f}  P(1|default)={self.p1_[1, j]:.3f}"
                             f"   evidence={self.feature_importances_[j]:.3f}")
            else:
                lines.append(f"{self.feature_names_[j]:<24} Categorical ({len(self.edges_[j]) + 1} bins)   evidence={self.feature_importances_[j]:.3f}")
        return "\n".join(lines)

    def feature_importance_table(self, top=None):
        order = np.argsort(-self.feature_importances_)
        rows = [(self.feature_names_[j], float(self.feature_importances_[j])) for j in order]
        return rows[:top] if top else rows

    def pick_threshold(self, X_val, y_val, metric="f1"):
        """Choose the decision threshold on the VALIDATION set; stored in best_threshold_."""
        p = self.predict_proba(X_val)[:, 1]
        y = np.asarray(y_val).astype(int)
        grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
        best_t, best_s = 0.5, -1.0
        for t in grid:
            pr = (p >= t).astype(int)
            tp = ((pr == 1) & (y == 1)).sum(); fp = ((pr == 1) & (y == 0)).sum()
            fn = ((pr == 0) & (y == 1)).sum(); tn = ((pr == 0) & (y == 0)).sum()
            if metric == "f1":
                s = 2 * tp / max(2 * tp + fp + fn, 1)
            else:                                           # balanced accuracy
                s = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
            if s > best_s:
                best_t, best_s = float(t), float(s)
        self.best_threshold_, self.best_threshold_score_ = best_t, best_s
        return best_t

    # --------------------------------------------------------- save / load
    def get_state(self):
        return {"kind": self.kind, "var_smoothing": self.var_smoothing, "alpha": self.alpha, "n_bins": self.n_bins,
                "prior": self.prior if isinstance(self.prior, str) else list(self.prior),
                "feature_names": self.feature_names_, "feature_type": self.feature_type_.tolist(),
                "class_prior": self.class_prior_.tolist(), "empirical_prior": self.empirical_prior_.tolist(),
                "theta": self.theta_, "var": self.var_, "p1": self.p1_, "epsilon": float(self.epsilon_),
                "edges": [None if e is None else e for e in self.edges_],
                "cat_logp": [None if c is None else c for c in self.cat_logp_],
                "importances": self.feature_importances_, "best_threshold": float(self.best_threshold_)}

    @classmethod
    def from_state(cls, s):
        m = cls(s["kind"], s["var_smoothing"], s["alpha"], s["n_bins"], s["prior"])
        m.feature_names_ = list(s["feature_names"]); m.feature_type_ = np.array(s["feature_type"])
        m.n_features_ = len(m.feature_names_); m.classes_ = np.array([0, 1])
        m.class_prior_ = np.array(s["class_prior"]); m.empirical_prior_ = np.array(s["empirical_prior"])
        m.theta_, m.var_, m.p1_, m.epsilon_ = s["theta"], s["var"], s["p1"], s["epsilon"]
        m.edges_, m.cat_logp_ = s["edges"], s["cat_logp"]
        m.feature_importances_, m.best_threshold_ = s["importances"], s["best_threshold"]
        return m

    def save(self, path):
        import joblib
        joblib.dump({"model": "NaiveBayesScratch", "state": self.get_state()}, path)

    @classmethod
    def load(cls, path):
        import joblib
        return cls.from_state(joblib.load(path)["state"])
