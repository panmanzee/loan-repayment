"""
Blend of an Explainable Boosting Machine and a bagged MLP  (the "better model", from scratch, NumPy only)

Why this model
--------------
On this data every sensible model lands at ROC-AUC ~0.67-0.69, so one model is rarely clearly better than another.
What *does* work a little is combining models that make different mistakes:

  * the EBM  (ExplainableBoosting.py)  is additive: a smooth, nearly linear score built one feature at a time.
    Low variance, but it cannot see interactions between features.
  * the MLP  (MLP.py)                  is neural: it can combine features, but one network is noisy
    (its result changes with the random initialisation and with the mini-batch order).

Two classic variance-reduction ideas are used on top of each other:

  1. BAGGING of the MLP: train `n_mlps` networks that differ only in their random seed (initial weights, dropout
     masks, batch order) and average their probabilities. The average has lower variance than any single network.
     Each network early-stops on the validation set.
  2. BLENDING (fixed-weight averaging) of the two *families*: p = w * p_EBM + (1 - w) * p_MLPbag, w = 0.5.
     No second-level model is trained, so unlike stacking nothing can over-fit the blend weights.

Both members are trained with the same sqrt class weight, so their probabilities live on the same scale and a plain
average keeps the probabilities usable (the log-loss stays meaningful).  The threshold is chosen on validation.

Interpretability is kept: `explain(x)` returns the EBM part exactly (each feature's contribution in log-odds) and the
MLP part as the difference it adds, so a decision can still be justified with reason codes.

Measured (8 random splits, test ROC-AUC): see docs and run_classification.py.  Honest summary: +0.003 AUC over
logistic regression and +0.006 over Random Forest, in 7 of 8 splits.  Small, but consistent, and the best PR-AUC.
"""
import time

import numpy as np

try:  # works both as a package import and when the folder is on sys.path (as run_classification.py does)
    from ExplainableBoosting import ExplainableBoostingScratch
    from MLP import MLPClassifierScratch
except ImportError:  # pragma: no cover
    from .ExplainableBoosting import ExplainableBoostingScratch
    from .MLP import MLPClassifierScratch


class BlendEnsembleScratch:
    def __init__(self, n_mlps=5, ebm_weight=0.5, mlp_params=None, ebm_params=None, random_state=0):
        self.n_mlps = n_mlps
        self.ebm_weight = ebm_weight
        self.mlp_params = dict(hidden_layers=(64, 32), l2=1e-2, dropout=0.2, learning_rate=1e-3, n_epochs=200)
        self.mlp_params.update(mlp_params or {})
        self.ebm_params = dict(n_rounds=800, learning_rate=0.02, max_bins=16, max_leaves=2, min_samples_leaf=60,
                               reg_lambda=20.0, n_outer_bags=8, n_interactions=0, linear_base=True)
        self.ebm_params.update(ebm_params or {})
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, eval_set=None, feature_names=None, early_stopping_rounds=20, sample_weight=None, verbose=False):
        t0 = time.time()
        kw = dict(eval_set=eval_set, feature_names=feature_names)
        self.ebm_ = ExplainableBoostingScratch(random_state=self.random_state, **self.ebm_params).fit(
            X, y, early_stopping_rounds=max(early_stopping_rounds, 40), **kw)
        self.mlps_ = [MLPClassifierScratch(random_state=self.random_state + k, **self.mlp_params).fit(
            X, y, early_stopping_rounds=early_stopping_rounds, **kw) for k in range(self.n_mlps)]
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(X.shape[1])]
        mlp_imp = np.mean([m.feature_importances_ for m in self.mlps_], axis=0)
        imp = self.ebm_weight * self.ebm_.feature_importances_ + (1 - self.ebm_weight) * mlp_imp
        self.feature_importances_ = imp / max(imp.sum(), 1e-12)
        # average of the members' unweighted validation log-loss curves, padded to the longest (for the performance chart)
        hs = [m.history_ for m in self.mlps_]
        L = max(len(h["val_logloss"]) for h in hs) if eval_set is not None else 0
        pad = lambda v: list(v) + [v[-1]] * (L - len(v))
        self.history_ = {"train_logloss": list(np.mean([pad(h["train_logloss"]) for h in hs], axis=0)) if L else [],
                         "val_logloss": list(np.mean([pad(h["val_logloss"]) for h in hs], axis=0)) if L else []}
        self.best_threshold_ = 0.5
        self.fit_time_ = time.time() - t0
        return self

    # ------------------------------------------------------- plug-in names used by run_classification.py
    @property
    def train_loss(self):
        return self.history_.get("train_logloss") or None

    @property
    def evals_result_(self):
        return {"validation": {"logloss": self.history_.get("val_logloss", [])}}

    # -------------------------------------------------------------- predict
    def member_probas(self, X):
        """(p_ebm, p_mlp_bag): the two halves of the blend."""
        return (self.ebm_.predict_proba(X)[:, 1],
                np.mean([m.predict_proba(X)[:, 1] for m in self.mlps_], axis=0))

    def predict_proba(self, X):
        p_e, p_m = self.member_probas(X)
        p = self.ebm_weight * p_e + (1 - self.ebm_weight) * p_m
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=None):
        thr = self.best_threshold_ if threshold is None else threshold
        return (self.predict_proba(X)[:, 1] >= thr).astype(int)

    def explain(self, X, top=6):
        """EBM reason codes for one borrower + how far the neural half moves the final probability."""
        x = np.asarray(X, dtype=float).reshape(1, -1)
        e = self.ebm_.explain(x, top=top)
        p_e, p_m = self.member_probas(x)
        e["p_ebm"], e["p_mlp_bag"], e["p_blend"] = float(p_e[0]), float(p_m[0]), float(self.predict_proba(x)[0, 1])
        return e

    # ----------------------------------------------------------- save / load
    def get_state(self):
        return {"cls": "BlendEnsembleScratch", "n_mlps": self.n_mlps, "ebm_weight": self.ebm_weight, "random_state": self.random_state,
                "mlp_params": self.mlp_params, "ebm_params": self.ebm_params, "ebm": self.ebm_.get_state(),
                "mlps": [m.get_state() for m in self.mlps_], "feature_names": self.feature_names_,
                "importances": self.feature_importances_, "best_threshold": float(self.best_threshold_), "history": self.history_}

    @classmethod
    def from_state(cls, s):
        m = cls(n_mlps=s["n_mlps"], ebm_weight=s["ebm_weight"], mlp_params=s["mlp_params"], ebm_params=s["ebm_params"], random_state=s["random_state"])
        m.ebm_ = ExplainableBoostingScratch.from_state(s["ebm"]); m.mlps_ = [MLPClassifierScratch.from_state(t) for t in s["mlps"]]
        m.feature_names_, m.feature_importances_, m.best_threshold_, m.history_ = s["feature_names"], s["importances"], s["best_threshold"], s["history"]
        return m

    def save(self, path):
        import joblib
        joblib.dump({"model": "BlendEnsembleScratch", "state": self.get_state()}, path)

    @classmethod
    def load(cls, path):
        import joblib
        return cls.from_state(joblib.load(path)["state"])
