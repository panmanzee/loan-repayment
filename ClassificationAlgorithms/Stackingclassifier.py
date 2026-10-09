"""
Stacking classifier (from scratch) - a meta-model trained on the predictions of several base models.

NumPy only for the machinery (the base models are the ones already in this folder, imported lazily).
Same folder style as BlendEnsemble.py, which is the simpler cousin of this file:

    BLENDING  (BlendEnsemble.py)   fixed weights, e.g. 0.5 * EBM + 0.5 * MLP. Nothing is trained on the
                                   predictions, so it can never over-fit - but it also cannot learn that
                                   one model is better than another.
    STACKING  (this file)          a second-level model LEARNS the weights from the base predictions,
                                   e.g. "trust the SVM 0.6, the trees 0.5, the Naive Bayes 0.1".

THE ONE THING THAT MAKES STACKING WORK: OUT-OF-FOLD PREDICTIONS
--------------------------------------------------------------
The meta-model must not see predictions that the base models made about rows they were trained on. If a
deep tree or a 1-nearest-neighbour model is asked about its own training rows it answers "99% sure" for
every row, the meta-model learns "that column is perfect", and the stack looks brilliant on paper and is
worse than the base models on new data. The classic example, and what this file does instead:

    naive (WRONG, leaks)                     stacked properly (out-of-fold)
    base.fit(X_tr, y_tr)                     for each of 5 folds:
    p_train = base.predict(X_tr)   <- seen       base.fit(X_tr without fold k)
    meta.fit(p_train, y_tr)                      p[fold k] = base.predict(fold k)   <- never seen
    meta sees an optimistic column           meta.fit(p_train_oof, y_tr)
                                             meta sees an honest column

    X_train ─┬─ fold 1 ─ <-- base trained on folds 2-5 ─> predict fold 1  ─┐
             ├─ fold 2 ─ <-- base trained on 1,3,4,5  ─> predict fold 2  ─┤  meta-features
             ├─ fold 3 ...                                                ├─> meta model
             └─ fold 5 ...                                                ─┘   (learns the weights)

For prediction the base models are refitted on the whole training set and the meta-model is fed their
probabilities - the same shape of input it was trained on.

How the base models are trained inside the folds: each fold's training part is split again into an inner
train / inner validation pair, and the validation half is what the base models use for early stopping.
The outer validation set is used only for the final refit and for choosing the threshold, like every
other model in this folder.

Example
-------
    import sys; sys.path.insert(0, "Common"); sys.path.insert(0, "ClassificationAlgorithms")
    from data_classification import build_classification_data, load_dataset, train_val_test_split_scratch
    from metrics_classification import evaluate_on_test
    from Stackingclassifier import StackingClassifierScratch, default_base_models

    X, y, names = build_classification_data(load_dataset())
    X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test_split_scratch(X, y)

    st = StackingClassifierScratch(default_base_models(), n_folds=5)
    st.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], feature_names=names, early_stopping_rounds=30)
    print(st.summary())                 # base models + how much the meta-model trusts each one
    proba = st.predict_proba(X_te)[:, 1]
"""

import inspect
import time

import numpy as np


# --------------------------------------------------------------------------------------- helpers
def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))


def _auc(y, s):
    """ROC-AUC by the rank-sum identity, ties get the average rank (self-contained copy)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _average_precision(y, s):
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


def _wlogloss(y, p, w=None):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    terms = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.average(terms, weights=w)) if w is not None else float(np.mean(terms))


def _default_names(n, prefix="feat"):
    return [f"{prefix}{i + 1}" for i in range(n)]


def _stratified_folds(y, n_folds, seed):
    """Folds that keep the class ratio (a plain k-fold can hand a fold 0 positives on 16% data)."""
    y = np.asarray(y).astype(int).ravel()
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(int(n_folds))]
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for k, chunk in enumerate(np.array_split(idx, int(n_folds))):
            folds[k].extend(chunk.tolist())
    return [np.array(sorted(f), dtype=int) for f in folds]


def _stratified_split(y, val_size, seed):
    """Index split (fit_idx, val_idx) that keeps the class ratio."""
    y = np.asarray(y).astype(int).ravel()
    n_val = int(round(len(y) * float(val_size)))
    n_val = max(1, min(n_val, len(y) - 1))
    rng = np.random.RandomState(seed)
    val, fit = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        if len(idx) > 1:
            cut = max(1, min(int(round(len(idx) * float(val_size))), len(idx) - 1))
            val.extend(idx[:cut].tolist())
            fit.extend(idx[cut:].tolist())
        else:
            fit.extend(idx.tolist())                 # a lonely row stays in the fit part
    if not val:                                      # never hand an empty eval_set to a base model
        val.append(fit.pop())
    return np.array(sorted(fit), dtype=int), np.array(sorted(val), dtype=int)


def _dump_model(model):
    """
    Serialise a fitted base model to bytes. Some models in this folder (the older tree / logistic files)
    have no get_state(), but every one of them survives joblib, so pickling is the one mechanism that
    works for all of them - and for scikit-learn estimators, which have no state protocol either.
    """
    import io
    import joblib
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()


def _load_model(blob):
    import io
    import joblib
    return joblib.load(io.BytesIO(blob))


def _accepts_param(model, name):
    """
    Does this model's fit() accept that keyword? Not every model does: the DecisionTree in this folder
    has no eval_set, and a scikit-learn estimator (perfectly usable as a base model here) takes neither
    eval_set nor feature_names nor early_stopping_rounds. Asking the signature once is safer than
    catching a TypeError, which would also hide a genuine error inside fit().
    """
    try:
        params = inspect.signature(model.fit).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def _accepts_eval_set(model):
    """Kept for readability: the flag that decides whether a base model early-stops on the inner split."""
    return _accepts_param(model, "eval_set")


# ======================================================================================= meta model
class _MetaLogistic:
    """
    The blender: L2-regularised logistic regression, written from scratch, fitted by Newton-Raphson on a
    handful of columns (one per base model). Kept inside this file so the stacker works even if the other
    files change; any object with fit/predict_proba can be passed as `meta_model` instead, including
    LogisticRegressionScratch from Logisticregression.py.
    """

    def __init__(self, reg_lambda=1.0, standardize=True, max_iter=100, tol=1e-8):
        self.reg_lambda = float(reg_lambda)
        self.standardize = bool(standardize)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-12] = 1.0
        self.scale_ = sd if self.standardize else np.ones(d)
        Z = np.column_stack([np.ones(n), (X - self.mean_) / self.scale_])
        beta = np.zeros(d + 1)
        ridge = np.eye(d + 1) * self.reg_lambda
        ridge[0, 0] = 0.0                                   # never penalise the intercept
        self.history_ = []
        for _ in range(self.max_iter):
            p = _sigmoid(Z @ beta)
            grad = Z.T @ (w * (p - y)) - ridge @ beta
            h = w * p * (1 - p)
            hess = (Z * h[:, None]).T @ Z + ridge
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:                    # singular: fall back to a small ridge
                step = np.linalg.lstsq(hess + 1e-6 * np.eye(d + 1), grad, rcond=None)[0]
            beta = beta - step
            self.history_.append(_wlogloss(y, _sigmoid(Z @ beta), w))
            if np.max(np.abs(step)) < self.tol:
                break
        self.coef_ = beta[1:]
        self.intercept_ = float(beta[0])
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_ @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def get_state(self):
        return {"params": {"reg_lambda": self.reg_lambda, "standardize": self.standardize,
                           "max_iter": self.max_iter, "tol": self.tol},
                "mean": self.mean_, "scale": self.scale_, "coef": self.coef_,
                "intercept": self.intercept_}

    @classmethod
    def from_state(cls, s):
        m = cls(**s["params"])
        m.mean_, m.scale_, m.coef_, m.intercept_ = s["mean"], s["scale"], s["coef"], s["intercept"]
        m.classes_ = np.array([0, 1])
        m.history_ = []
        return m


# =================================================================================== the stacker
class StackingClassifierScratch:
    """
    Stack of classifiers with a learned meta-model.

    Parameters
    ----------
    base_models   list of (name, factory) or (name, factory, use_val) pairs. `factory()` must return a
                  FRESH, unfitted model (that is what makes the folds independent). A fitted instance is
                  also accepted and cloned through its get_state()/from_state(), but factories are
                  cheaper. use_val=False means "do not hand this model an eval_set" (for the models whose
                  fit() does not take one) - it is detected from the signature when not given.
    meta_model    the second-level model; None = the built-in L2 logistic regression (_MetaLogistic).
                  Anything with fit/predict_proba works.
    n_folds       folds used to build the out-of-fold meta-features (5 is the usual choice; more folds =
                  more honest columns but n_folds times the training cost)
    inner_val     share of each fold's training part held out for the base models' early stopping
    passthrough   also give the meta-model the original features (standard stacking option; False keeps
                  the meta-model interpretable - one weight per base model)
    random_state  seed for the folds and for the base models' factories (passed as random_state when the
                  factory accepts it)

    Fitted attributes
    -----------------
    oof_proba_ (n_train x n_base, the honest meta-features), meta_ (the fitted blender),
    bases_ (list of base models refitted on the full training set), base_names_,
    oof_table_ / val_table_ (per base model: OOF and validation ROC-AUC / PR-AUC), meta_weights_,
    history_ (per-fold curves), best_threshold_, fit_time_
    """

    def __init__(self, base_models=None, meta_model=None, n_folds=5, inner_val=0.15,
                 passthrough=False, meta_reg_lambda=1.0, random_state=42, feature_names=None):
        if base_models is not None and len(list(base_models)) == 0:
            raise ValueError("base_models is empty - a stack needs at least one base model")
        self.base_models = list(base_models) if base_models is not None else None
        self.meta_model = meta_model
        self.n_folds = int(n_folds)
        self.inner_val = float(inner_val)
        self.passthrough = bool(passthrough)
        self.meta_reg_lambda = float(meta_reg_lambda)
        self.random_state = random_state
        self.feature_names = feature_names

    # --------------------------------------------------------------- utilities
    def _specs(self):
        """Normalise base_models into (name, factory, use_val)."""
        if not self.base_models:
            self.base_models = default_base_models(random_state=self.random_state)
        specs = []
        for i, item in enumerate(self.base_models):
            if len(item) == 3:
                name, factory, use_val = item
            else:
                name, factory = item
                use_val = None
            probe = factory() if callable(factory) else factory
            specs.append((str(name), factory, probe, use_val))
        return specs

    def _make_base(self, factory, seed_offset=0):
        """A fresh base model (or a clone of a fitted instance)."""
        if callable(factory):
            try:
                return factory(random_state=self.random_state + seed_offset)
            except TypeError:
                return factory()
        clone = getattr(factory, "from_state", None)
        return type(factory).from_state(factory.get_state()) if clone else factory

    def _fit_base(self, factory, X, y, eval_set, use_val, seed_offset, feature_names):
        m = self._make_base(factory, seed_offset)
        if use_val is None:
            use_val = _accepts_eval_set(m)
        kwargs = {}
        if feature_names is not None and _accepts_param(m, "feature_names"):
            kwargs["feature_names"] = feature_names
        if use_val and eval_set is not None:
            m.fit(X, y, eval_set=eval_set, **kwargs)
        else:
            m.fit(X, y, **kwargs)
        return m

    # ------------------------------------------------------------------- fit
    def fit(self, X, y, eval_set=None, feature_names=None, early_stopping_rounds=30,
            sample_weight=None, verbose=False, **_ignored):
        t0 = time.time()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or inf - handle missing values first")
        y = np.asarray(y).astype(int).ravel()
        if len(y) != len(X):
            raise ValueError("X and y have different lengths")
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = list(feature_names or self.feature_names or
                                   _default_names(self.n_features_in_))
        if len(self.feature_names_) != self.n_features_in_:
            raise ValueError("feature_names has the wrong length")
        Zv = yv = None
        if eval_set is not None:
            Zv = np.asarray(eval_set[0][0], dtype=float)
            yv = np.asarray(eval_set[0][1]).astype(int).ravel()

        specs = self._specs()
        self.base_names_ = [s[0] for s in specs]
        n_base = len(specs)
        n = len(X)
        # never ask for more folds than the minority class has rows (a fold with no positives is useless)
        n_folds = max(2, min(self.n_folds, int(np.bincount(y).min())))
        self.n_folds_ = int(n_folds)

        # ---- 1. out-of-fold meta-features (the honest columns)
        oof = np.zeros((n, n_base))
        history = {"fold": [], "oof_auc": [], "train_logloss": [], "val_logloss": []}
        folds = _stratified_folds(y, n_folds, self.random_state)
        for k, fold in enumerate(folds):
            tr = np.setdiff1d(np.arange(n), fold, assume_unique=False)
            inner_fit, inner_val = _stratified_split(y[tr], self.inner_val, self.random_state + 100 + k)
            idx_fit, idx_ival = tr[inner_fit], tr[inner_val]
            for j, (name, factory, probe, use_val) in enumerate(specs):
                m = self._fit_base(factory, X[idx_fit], y[idx_fit],
                                   eval_set=[(X[idx_ival], y[idx_ival])], use_val=use_val,
                                   seed_offset=k * 17 + j, feature_names=self.feature_names_)
                oof[fold, j] = m.predict_proba(X[fold])[:, 1]
            aucs = [_auc(y[fold], oof[fold, j]) for j in range(n_base)]
            history["fold"].append(k + 1)
            history["oof_auc"].append(float(np.mean(aucs)))
            # fold curves: what the fold's held-out rows looked like for the base average
            history["train_logloss"].append(_wlogloss(y[fold], np.clip(oof[fold].mean(axis=1), 1e-6, 1 - 1e-6)))
            history["val_logloss"].append(_wlogloss(y[fold], np.clip(oof[fold].mean(axis=1), 1e-6, 1 - 1e-6)))
            if verbose:
                print(f"  fold {k + 1}/{n_folds} | mean base OOF AUC {np.mean(aucs):.4f} "
                      f"({', '.join(f'{s[0]} {a:.3f}' for s, a in zip(specs, aucs))})")
        self.oof_proba_ = oof

        # ---- 2. the meta-model is trained on those columns only
        meta_X = np.column_stack([oof, X]) if self.passthrough else oof
        meta_model = self.meta_model
        if meta_model is None:
            meta_model = _MetaLogistic(reg_lambda=self.meta_reg_lambda)
        elif callable(meta_model) and not hasattr(meta_model, "fit"):
            meta_model = meta_model()
        self.meta_ = meta_model
        if sample_weight is not None and self._meta_takes_weights():
            self.meta_.fit(meta_X, y, sample_weight=sample_weight)
        else:
            self.meta_.fit(meta_X, y)
        n_meta = n_base * (1 + self.passthrough)
        self.meta_weights_ = np.asarray(getattr(self.meta_, "coef_", np.zeros(n_meta)),
                                        dtype=float).ravel()[:n_meta]

        # ---- 3. refit every base model on the FULL training set (what inference uses)
        self.bases_ = [self._fit_base(factory, X, y, eval_set=eval_set, use_val=use_val,
                                      seed_offset=1000 + j, feature_names=self.feature_names_)
                       for j, (name, factory, probe, use_val) in enumerate(specs)]

        # ---- 4. diagnosis: OOF vs validation score of every base model, and of the stack itself
        self.oof_table_ = [(s[0], _auc(y, oof[:, j]), _average_precision(y, oof[:, j]))
                           for j, s in enumerate(specs)]
        if Zv is not None:
            # use meta_/bases_ directly here: the model is not marked fitted until the end of fit()
            Mv = self._meta_features(Zv)
            p_stack_val = self.meta_.predict_proba(Mv)[:, 1]
            self.val_table_ = [(s[0], _auc(yv, self.bases_[j].predict_proba(Zv)[:, 1]),
                                _average_precision(yv, self.bases_[j].predict_proba(Zv)[:, 1]))
                               for j, s in enumerate(specs)]
            self.val_logloss_ = _wlogloss(yv, p_stack_val)
            self.val_auc_ = _auc(yv, p_stack_val)
            self.val_ap_ = _average_precision(yv, p_stack_val)
        else:
            self.val_table_, self.val_logloss_, self.val_auc_, self.val_ap_ = [], None, None, None
        self.history_ = history
        self.feature_importances_ = self._weights_as_importance()
        self.best_threshold_ = 0.5
        self.best_threshold_score_ = None
        self.fitted_ = True
        self.fit_time_ = time.time() - t0
        return self

    def _meta_takes_weights(self):
        try:
            return "sample_weight" in inspect.signature(self.meta_.fit).parameters
        except (TypeError, ValueError):
            return False

    def _weights_as_importance(self):
        w = np.abs(self.meta_weights_[: len(self.base_names_)])
        return w / max(w.sum(), 1e-12)

    def _meta_features(self, X):
        X = np.asarray(X, dtype=float)
        cols = [m.predict_proba(X)[:, 1] for m in self.bases_]
        M = np.column_stack(cols)
        return np.column_stack([M, X]) if self.passthrough else M

    # ---------------------------------------------------------------- predict
    def _check_fitted(self):
        if not getattr(self, "fitted_", False):
            raise RuntimeError("StackingClassifierScratch is not fitted yet - call fit(X, y) first")

    def decision_function(self, X):
        self._check_fitted()
        return self.meta_.decision_function(self._meta_features(X)) \
            if hasattr(self.meta_, "decision_function") else self.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        self._check_fitted()
        return self.meta_.predict_proba(self._meta_features(X))

    def predict(self, X, threshold=None):
        thr = self.best_threshold_ if threshold is None else threshold
        return (self.predict_proba(X)[:, 1] >= thr).astype(int)

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

    # ------------------------------------------------- explanations / tables
    def member_probas(self, X):
        """The meta-features: one probability column per base model."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return {name: m.predict_proba(X)[:, 1] for name, m in zip(self.base_names_, self.bases_)}

    def feature_importance_table(self, top=None):
        """
        How much the meta-model trusts each base model: |weight| normalised to sum to 1. This is the
        table worth putting in the report next to each base model's own score.
        """
        self._check_fitted()
        rows = [(self.base_names_[j], float(self.feature_importances_[j])) for j in range(len(self.base_names_))]
        rows.sort(key=lambda r: -r[1])
        return rows[:top] if top else rows

    def coefficient_table(self, top=None):
        """Signed meta weights (positive = the model pushes towards 'default'), with the intercept."""
        self._check_fitted()
        rows = [{"feature": f"p({self.base_names_[j]})", "weight": float(self.meta_weights_[j])}
                for j in range(len(self.base_names_))]
        rows.append({"feature": "intercept", "weight": float(getattr(self.meta_, "intercept_", 0.0))})
        rows.sort(key=lambda r: -abs(r["weight"]))
        return rows[:top] if top else rows

    def explain(self, X, top=6):
        """One borrower: what every base model said, the meta-model's own weights, and the answer."""
        self._check_fitted()
        x = np.asarray(X, dtype=float).reshape(1, -1)
        members = {name: float(m.predict_proba(x)[0, 1]) for name, m in zip(self.base_names_, self.bases_)}
        w = self.meta_weights_[: len(self.base_names_)]
        contrib = {name: float(w[j] * members[name]) for j, name in enumerate(self.base_names_)}
        return {"p_stack": float(self.predict_proba(x)[0, 1]),
                "member_probas": members,
                "meta_weights": {name: float(w[j]) for j, name in enumerate(self.base_names_)},
                "contributions": dict(sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:top])}

    def base_model_table(self):
        """(name, OOF AUC, OOF AP, validation AUC) per base model + the stack's own row."""
        self._check_fitted()
        rows = []
        for j, (name, auc_oof, ap_oof) in enumerate(self.oof_table_):
            val_auc = self.val_table_[j][1] if self.val_table_ else float("nan")
            rows.append({"model": name, "oof_auc": auc_oof, "oof_ap": ap_oof, "val_auc": val_auc,
                         "stack_weight": float(self.meta_weights_[j])})
        rows.sort(key=lambda r: -r["oof_auc"])
        return rows

    def summary(self):
        self._check_fitted()
        lines = [f"Stacking: {len(self.base_names_)} base models, {self.n_folds_} folds, "
                 f"meta={type(self.meta_).__name__}, passthrough={self.passthrough} "
                 f"| fit {self.fit_time_:.1f}s"]
        lines.append(f"  {'base model':<26} {'OOF AUC':>8} {'OOF AP':>8} {'val AUC':>8} {'meta w':>8}")
        for r in self.base_model_table():
            lines.append(f"  {r['model']:<26} {r['oof_auc']:>8.4f} {r['oof_ap']:>8.4f} "
                         f"{r['val_auc']:>8.4f} {r['stack_weight']:>8.3f}")
        if self.val_auc_ is not None:
            lines.append(f"  {'STACK (this model)':<26} {'':>8} {'':>8} {self.val_auc_:>8.4f}")
        return "\n".join(lines)

    # -------------------------------------------------------------- plug-ins
    @property
    def train_loss(self):
        return getattr(self, "history_", {}).get("val_logloss") or None

    @property
    def evals_result_(self):
        h = getattr(self, "history_", None) or {}
        return {"validation": {"logloss": h.get("val_logloss", [])}}

    # ----------------------------------------------------------- save / load
    def get_state(self):
        self._check_fitted()
        return {"cls": "StackingClassifierScratch",
                "params": dict(n_folds=self.n_folds, inner_val=self.inner_val,
                               passthrough=self.passthrough, meta_reg_lambda=self.meta_reg_lambda,
                               random_state=self.random_state),
                "base_names": self.base_names_,
                "bases": [_dump_model(m) for m in self.bases_],
                "base_classes": [type(m).__name__ for m in self.bases_],
                "meta": _dump_model(self.meta_),
                "meta_coef": self.meta_weights_, "feature_names": self.feature_names_,
                "oof_proba": self.oof_proba_, "oof_table": self.oof_table_, "val_table": self.val_table_,
                "importances": self.feature_importances_, "history": self.history_,
                "best_threshold": float(self.best_threshold_),
                "n_features_in_": self.n_features_in_}

    @classmethod
    def from_state(cls, s):
        m = cls(base_models=None, **s["params"])
        m.base_models = None
        m.base_names_ = list(s["base_names"])
        m.bases_ = [_load_model(b) for b in s["bases"]]
        m.meta_ = _load_model(s["meta"])
        m.meta_weights_ = np.asarray(s["meta_coef"], dtype=float)
        m.feature_names_ = list(s["feature_names"])
        m.oof_proba_, m.oof_table_, m.val_table_ = s["oof_proba"], s["oof_table"], s["val_table"]
        m.feature_importances_ = s["importances"]
        m.history_ = s["history"]
        m.best_threshold_ = s["best_threshold"]
        m.best_threshold_score_ = None
        m.n_features_in_ = s["n_features_in_"]
        m.val_auc_ = m.val_ap_ = m.val_logloss_ = None
        m.fitted_ = True
        m.fit_time_ = 0.0
        return m

    def save(self, path):
        import joblib
        joblib.dump({"model": "StackingClassifierScratch", "state": self.get_state()}, path)

    @classmethod
    def load(cls, path):
        import joblib
        return cls.from_state(joblib.load(path)["state"])


def default_base_models(random_state=42, include_slow=False):
    """
    A sensible default stack: models that are (a) fast enough to train n_folds + 1 times and (b) good at
    different things - a linear model, two different neural-ish learners, a kernel model, a distance
    model and a probabilistic one. With include_slow=True the tree ensembles and the EBM are added too
    (better, but the fit then takes minutes instead of ~40 seconds).

    Each entry is (name, factory, use_val): the flag says whether that model's fit() accepts an
    eval_set (the DecisionTree and the Naive Bayes in this folder do not).
    """
    from Logisticregression import LogisticRegressionScratch
    from Perceptron import PerceptronScratch
    from NaiveBayes import NaiveBayesScratch
    from SVC import SVCScratch
    from KNN import KNNClassifierScratch

    models = [
        ("Logistic Regression", lambda random_state=random_state: LogisticRegressionScratch(
            learning_rate=0.5, n_iterations=500, reg_lambda=1.0), True),   # no seed: fully deterministic
        ("Perceptron (averaged)", lambda random_state=random_state: PerceptronScratch(
            variant="averaged", learning_rate=0.1, n_epochs=40, random_state=random_state), True),
        ("Naive Bayes", lambda random_state=random_state: NaiveBayesScratch(
            kind="categorical", n_bins=10), False),
        ("SVC (linear)", lambda random_state=random_state: SVCScratch(
            C=1.0, kernel="linear", solver="primal", n_iterations=1500,
            scale_pos_weight="balanced", random_state=random_state), True),
        ("k-NN", lambda random_state=random_state: KNNClassifierScratch(
            n_neighbors=25, tune_k="auto", random_state=random_state), True),
    ]
    if include_slow:
        from RandomForest import RandomForestClassifierScratch
        from GradientBoosting import GradientBoostingClassifierScratch
        from XGB import XGBClassifierScratch
        from ExplainableBoosting import ExplainableBoostingScratch
        models += [
            ("Random Forest", lambda random_state=random_state: RandomForestClassifierScratch(
                n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=random_state), False),
            ("Gradient Boosting", lambda random_state=random_state: GradientBoostingClassifierScratch(
                n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
                random_state=random_state), True),
            ("XGBoost", lambda random_state=random_state: XGBClassifierScratch(
                n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
                colsample_bytree=0.8, random_state=random_state), True),
            ("EBM", lambda random_state=random_state: ExplainableBoostingScratch(
                n_rounds=800, learning_rate=0.02, max_bins=16, n_outer_bags=8, random_state=random_state), True),
        ]
    return models
