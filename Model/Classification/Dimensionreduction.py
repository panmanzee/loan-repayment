"""PCA, Kernel PCA และ LDA สำหรับลดมิติข้อมูล"""

import time

import numpy as np


def _as_float(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or inf - handle missing values first")
    return X


def _default_names(n, prefix="feat"):
    return [f"{prefix}{i + 1}" for i in range(n)]


def _sign_fix(V):
    V = np.array(V, dtype=float)
    for j in range(V.shape[1]):
        k = int(np.argmax(np.abs(V[:, j])))
        if V[k, j] < 0:
            V[:, j] = -V[:, j]
    return V


class _TransformerBase:
    def _check_fitted(self):
        if not getattr(self, "fitted_", False):
            raise RuntimeError(f"{type(self).__name__} is not fitted yet - call fit(X[, y]) first")

    def _set_out_names(self, n_out):
        self.output_names_ = _default_names(n_out, getattr(self, "out_prefix_", "comp"))

    def get_state(self):
        self._check_fitted()
        keep = (
            "mean_",
            "scale_",
            "components_",
            "explained_variance_",
            "explained_variance_ratio_",
            "cumulative_variance_ratio_",
            "singular_values_",
            "loadings_",
            "scalings_",
            "class_means_",
            "priors_",
            "classes_",
            "discriminant_values_",
            "kernel_eigenvalues_",
            "kernel_eigenvectors_",
            "X_fit_",
            "_gamma",
            "_K_col_mean",
            "_K_mean",
            "reconstruction_error_",
            "fit_time_",
            "fitted_",
            "n_features_in_",
            "n_components_",
            "input_names_",
            "output_names_",
            "out_prefix_",
        )
        arrays = {k: getattr(self, k) for k in keep if hasattr(self, k)}
        return {"cls": type(self).__name__, "params": self._params(), "arrays": arrays}

    @classmethod
    def from_state(cls, state):
        m = cls(**state["params"])
        m.__dict__.update(state["arrays"])
        m.fitted_ = True
        return m

    def save(self, path):
        import joblib

        joblib.dump({"model": type(self).__name__, "state": self.get_state()}, path)

    @classmethod
    def load(cls, path):
        import joblib

        blob = joblib.load(path)
        if blob["model"] != cls.__name__:
            raise ValueError(f"file holds a {blob['model']}, not a {cls.__name__}")
        return cls.from_state(blob["state"])


class PCA(_TransformerBase):
    def __init__(self, n_components=None, scale=False, whiten=False, out_prefix_="PC"):
        self.n_components = n_components
        self.scale = bool(scale)
        self.whiten = bool(whiten)
        self.out_prefix_ = out_prefix_

    def _params(self):
        return {
            "n_components": self.n_components,
            "scale": self.scale,
            "whiten": self.whiten,
            "out_prefix_": self.out_prefix_,
        }

    def _resolve_k(self, eigvals):
        d = len(eigvals)
        n_comp = self.n_components
        if n_comp is None:
            return max(1, int(np.sum(eigvals > 1e-12)))
        if isinstance(n_comp, float) and 0 < n_comp < 1:
            ratio = np.cumsum(eigvals) / max(eigvals.sum(), 1e-12)
            return int(np.searchsorted(ratio, n_comp) + 1)
        k = int(n_comp)
        if k < 1 or k > d:
            raise ValueError(f"n_components must be in 1..{d} (or a float in (0,1)), got {n_comp}")
        return k

    def fit(self, X, y=None, feature_names=None):
        X = _as_float(X)
        n, d = X.shape
        t0 = time.time()
        self.n_features_in_ = d
        self.input_names_ = list(feature_names) if feature_names is not None else _default_names(d)

        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0) if self.scale else np.ones(d)
        self.scale_[self.scale_ < 1e-12] = 1.0
        Xc = (X - self.mean_) / self.scale_

        cov = (Xc.T @ Xc) / max(n - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(-eigvals)
        eigvals = np.clip(eigvals[order], 0.0, None)
        eigvecs = _sign_fix(eigvecs[:, order])

        k = self._resolve_k(eigvals)
        self.components_ = eigvecs[:, :k].T.copy()
        self.explained_variance_ = eigvals[:k].copy()
        total = float(eigvals.sum()) if eigvals.sum() > 0 else 1.0
        self.explained_variance_ratio_ = self.explained_variance_ / total
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
        self.singular_values_ = np.sqrt(self.explained_variance_ * max(n - 1, 1))
        self.n_components_ = k
        self._set_out_names(k)

        scores = Xc @ self.components_.T
        self.loadings_ = np.zeros((d, k))
        for j in range(d):
            if Xc[:, j].std() <= 1e-12:
                continue
            for c in range(k):
                if scores[:, c].std() > 1e-12:
                    self.loadings_[j, c] = np.corrcoef(Xc[:, j], scores[:, c])[0, 1]

        recon = (scores @ self.components_) * self.scale_ + self.mean_
        self.reconstruction_error_ = float(np.mean((X - recon) ** 2))
        self.fitted_ = True
        self.fit_time_ = time.time() - t0
        return self

    def transform(self, X):
        self._check_fitted()
        X = _as_float(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, the model was fitted on {self.n_features_in_}")
        Z = ((X - self.mean_) / self.scale_) @ self.components_.T
        if self.whiten:
            Z = Z / np.sqrt(np.maximum(self.explained_variance_, 1e-12))
        return Z

    def fit_transform(self, X, y=None, feature_names=None):
        return self.fit(X, y, feature_names=feature_names).transform(X)

    def inverse_transform_raw(self, Z):
        Z = np.asarray(Z, dtype=float)
        Z = Z[:, : self.n_components_]
        if self.whiten:
            Z = Z * np.sqrt(np.maximum(self.explained_variance_, 1e-12))
        return (Z @ self.components_) * self.scale_ + self.mean_

    def inverse_transform(self, Z):
        self._check_fitted()
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)
        return self.inverse_transform_raw(Z)

    def reconstruction_error(self, X):
        X = _as_float(X)
        return float(np.mean((X - self.inverse_transform(self.transform(X))) ** 2))

    def summary(self, top=5):
        self._check_fitted()
        lines = [
            f"PCA: {self.n_features_in_} features -> {self.n_components_} components | keeps "
            f"{self.cumulative_variance_ratio_[-1]:.2%} of the variance | reconstruction MSE "
            f"{self.reconstruction_error_:.4f}",
            "  variance per component: "
            + "  ".join(
                f"{self.output_names_[c]}={self.explained_variance_ratio_[c]:.1%}"
                for c in range(min(top, self.n_components_))
            ),
        ]
        for c in range(min(top, self.n_components_)):
            load = self.loadings_[:, c]
            idx = np.argsort(-np.abs(load))[:3]
            lines.append(
                f"  {self.output_names_[c]} driven by: "
                + ", ".join(f"{self.input_names_[j]} ({load[j]:+.2f})" for j in idx)
            )
        return "\n".join(lines)


class LDA(_TransformerBase):
    def __init__(self, n_components=None, shrinkage=0.0, scale=False, out_prefix_="LD"):
        self.n_components = n_components
        self.shrinkage = float(shrinkage)
        self.scale = bool(scale)
        self.out_prefix_ = out_prefix_

    def _params(self):
        return {
            "n_components": self.n_components,
            "shrinkage": self.shrinkage,
            "scale": self.scale,
            "out_prefix_": self.out_prefix_,
        }

    def fit(self, X, y, feature_names=None):
        X = _as_float(X)
        y = np.asarray(y).astype(int).ravel()
        if len(y) != len(X):
            raise ValueError("X and y have different lengths")
        n, d = X.shape
        t0 = time.time()
        self.n_features_in_ = d
        self.input_names_ = list(feature_names) if feature_names is not None else _default_names(d)
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("LDA needs at least two classes")
        self.classes_ = classes
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0) if self.scale else np.ones(d)
        self.scale_[self.scale_ < 1e-12] = 1.0
        X = (X - self.mean_) / self.scale_
        self.means_ = X.mean(axis=0)
        self.class_means_ = np.array([X[y == c].mean(axis=0) for c in classes])
        self.priors_ = np.array([float(np.mean(y == c)) for c in classes])

        Sw = np.zeros((d, d))
        for ci, c in enumerate(classes):
            Xc = X[y == c] - self.class_means_[ci]
            Sw += Xc.T @ Xc
        Sw /= max(n - len(classes), 1)
        if self.shrinkage > 0:
            Sw = (1.0 - self.shrinkage) * Sw + self.shrinkage * np.trace(Sw) / d * np.eye(d)

        Sb = np.zeros((d, d))
        for ci, c in enumerate(classes):
            diff = (self.class_means_[ci] - self.means_).reshape(-1, 1)
            Sb += self.priors_[ci] * (diff @ diff.T)

        w_eval, w_evec = np.linalg.eigh(Sw)
        w_eval = np.clip(w_eval, 1e-12, None)
        Wh = (w_evec / np.sqrt(w_eval)) @ w_evec.T
        M = Wh.T @ Sb @ Wh
        m_eval, m_evec = np.linalg.eigh(M)
        order = np.argsort(-m_eval)
        m_eval = np.clip(m_eval[order], 0.0, None)
        scalings = Wh @ m_evec[:, order]

        k_max = min(len(classes) - 1, d)
        k = k_max if self.n_components is None else max(1, min(int(self.n_components), k_max))
        self.scalings_ = _sign_fix(scalings[:, :k])
        self.discriminant_values_ = m_eval[:k]
        tot = float(m_eval[:k].sum()) if m_eval[:k].sum() > 0 else 1.0
        self.explained_variance_ratio_ = m_eval[:k] / tot
        self.n_components_ = k
        self._set_out_names(k)
        self.fitted_ = True
        self.fit_time_ = time.time() - t0
        return self

    def transform(self, X):
        self._check_fitted()
        X = _as_float(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, the model was fitted on {self.n_features_in_}")
        return ((X - self.mean_) / self.scale_ - self.means_) @ self.scalings_

    def fit_transform(self, X, y, feature_names=None):
        return self.fit(X, y, feature_names=feature_names).transform(X)

    def summary(self, top=3):
        self._check_fitted()
        lines = [
            f"LDA: {self.n_features_in_} features -> {self.n_components_} component(s) | "
            f"{self.explained_variance_ratio_[0]:.1%} of the class separation on the first axis"
        ]
        for c in range(min(top, self.n_components_)):
            s = self.scalings_[:, c]
            idx = np.argsort(-np.abs(s))[:3]
            lines.append(
                f"  {self.output_names_[c]} driven by: "
                + ", ".join(f"{self.input_names_[j]} ({s[j]:+.3f})" for j in idx)
            )
        return "\n".join(lines)


class KernelPCA(_TransformerBase):
    def __init__(
        self, n_components=2, kernel="rbf", gamma="scale", degree=3, coef0=1.0, max_samples=6000, out_prefix_="KPC"
    ):
        if kernel not in ("rbf", "poly", "linear"):
            raise ValueError('kernel must be "rbf", "poly" or "linear"')
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.max_samples = int(max_samples)
        self.out_prefix_ = out_prefix_

    def _params(self):
        return {
            "n_components": self.n_components,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "degree": self.degree,
            "coef0": self.coef0,
            "max_samples": self.max_samples,
            "out_prefix_": self.out_prefix_,
        }

    def _resolve_gamma(self, X):
        if self.gamma == "scale":
            return 1.0 / max(X.shape[1] * X.var(), 1e-12)
        if self.gamma == "auto":
            return 1.0 / max(X.shape[1], 1)
        return float(self.gamma)

    def _kernel(self, A, B):
        if self.kernel == "linear":
            return A @ B.T
        if self.kernel == "poly":
            return (self._gamma * (A @ B.T) + self.coef0) ** self.degree
        aa = np.sum(A**2, axis=1).reshape(-1, 1)
        bb = np.sum(B**2, axis=1).reshape(1, -1)
        return np.exp(-self._gamma * np.maximum(aa - 2.0 * (A @ B.T) + bb, 0.0))

    def fit(self, X, y=None, feature_names=None):
        X = _as_float(X)
        n, d = X.shape
        if n > self.max_samples:
            raise ValueError(
                f"KernelPCA needs an n x n kernel matrix; n={n} > max_samples={self.max_samples}. "
                f"Subsample, or use PCA (which is O(n*d))."
            )
        t0 = time.time()
        self.n_features_in_ = d
        self.input_names_ = list(feature_names) if feature_names is not None else _default_names(d)
        self._gamma = self._resolve_gamma(X)
        self.X_fit_ = X.copy()

        K = self._kernel(X, X)
        self._K_col_mean = K.mean(axis=0)
        self._K_mean = float(K.mean())
        Kc = K - self._K_col_mean.reshape(1, -1) - self._K_col_mean.reshape(-1, 1) + self._K_mean
        eigvals, eigvecs = np.linalg.eigh((Kc + Kc.T) / 2.0)
        order = np.argsort(-eigvals)
        eigvals = np.clip(eigvals[order], 0.0, None)
        eigvecs = _sign_fix(eigvecs[:, order])

        k = self.n_components
        if isinstance(k, float) and 0 < k < 1:
            ratio = np.cumsum(eigvals) / max(eigvals.sum(), 1e-12)
            k = int(np.searchsorted(ratio, k) + 1)
        k = max(1, min(int(k), n))
        self.n_components_ = k
        self.kernel_eigenvalues_ = eigvals[:k].copy()
        self.kernel_eigenvectors_ = eigvecs[:, :k].copy()
        tot = float(eigvals[:k].sum()) if eigvals[:k].sum() > 0 else 1.0
        self.explained_variance_ratio_ = eigvals[:k] / tot
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
        self._set_out_names(k)
        self.fitted_ = True
        self.fit_time_ = time.time() - t0
        return self

    def fit_transform(self, X, y=None, feature_names=None):
        self.fit(X, y, feature_names=feature_names)
        return self.kernel_eigenvectors_ * np.sqrt(np.maximum(self.kernel_eigenvalues_, 1e-12))

    def transform(self, X):
        self._check_fitted()
        X = _as_float(X)
        K_new = self._kernel(X, self.X_fit_)
        Kc = (
            K_new - K_new.mean(axis=1, keepdims=True) - self._K_col_mean.reshape(1, -1) + self._K_mean
        )
        return Kc @ (self.kernel_eigenvectors_ / np.sqrt(np.maximum(self.kernel_eigenvalues_, 1e-12)))

    def summary(self, top=3):
        self._check_fitted()
        return (
            f"KernelPCA({self.kernel}, gamma={self._gamma:.4g}): {self.n_features_in_} features -> "
            f"{self.n_components_} components | keeps {self.cumulative_variance_ratio_[-1]:.1%} of "
            f"the kernel variance"
        )


def feature_selection_by_importance(
    make_model,
    feature_names,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    metric=None,
    min_keep=5,
    tol=0.01,
    verbose=False,
):
    base = make_model().fit(np.asarray(X_train, dtype=float), np.asarray(y_train).astype(int))
    imp = getattr(base, "feature_importances_", None)
    if imp is None:
        raise ValueError(
            "the model has no feature_importances_ - use a tree model (RF / GB / XGB) "
            "for feature_selection_by_importance"
        )
    names = list(feature_names)
    imp = np.asarray(imp, dtype=float)
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train).astype(int)
    use_score = X_val is not None and y_val is not None and metric is not None
    X_val = np.asarray(X_val, dtype=float) if use_score else None
    y_val = np.asarray(y_val).astype(int) if use_score else None

    def evaluate(cols):
        m = make_model().fit(X_train[:, cols], y_train)
        if not use_score:
            return float("nan"), m
        return float(metric(y_val, m.predict_proba(X_val[:, cols])[:, 1])), m

    cols = list(np.argsort(-imp))
    score, model = evaluate(cols)
    best = (score, list(cols), model)
    scores = [(len(cols), score)]
    dropped = []
    if verbose:
        print(f"  {len(cols)} features: score {score:.4f}")
    while len(cols) > max(int(min_keep), 1):
        trial = cols[:-1]
        s, m = evaluate(trial)
        if use_score and s < score - tol:
            if verbose:
                print(
                    f"  dropping {names[cols[-1]]} would cost {score - s:.4f} (> tol {tol}) "
                    f"-> stop at {len(cols)} features"
                )
            break
        dropped.append(names[cols[-1]])
        cols, score, model = trial, s, m
        scores.append((len(cols), score))
        if s > best[0]:
            best = (s, list(cols), m)
        if verbose:
            print(f"  dropped {dropped[-1]:<26} -> {len(cols)} features, score {score:.4f}")
    keep_idx, best_model = best[1], best[2]
    return {
        "keep": [names[j] for j in keep_idx],
        "keep_idx": keep_idx,
        "dropped": dropped,
        "scores": scores,
        "model": best_model,
        "best_score": best[0],
    }


def compare_reductions(X_train, y_train, X_test, y_test, make_model, roc_auc, n_components=(2, 5, 10), verbose=True):
    out = {}
    m = make_model().fit(X_train, y_train)
    out["raw"] = float(roc_auc(y_test, m.predict_proba(X_test)[:, 1]))
    if verbose:
        print(f"  {'raw features':<20} {X_train.shape[1]:>3} features | test ROC-AUC {out['raw']:.4f}")
    for nc in n_components:
        p = PCA(n_components=nc, scale=True).fit(X_train)
        m = make_model().fit(p.transform(X_train), y_train)
        key = f"pca{nc}"
        out[key] = float(roc_auc(y_test, m.predict_proba(p.transform(X_test))[:, 1]))
        if verbose:
            print(
                f"  {('PCA ' + str(nc) + ' comp'):<20} {nc:>3} features | test ROC-AUC {out[key]:.4f}"
                f" | keeps {p.cumulative_variance_ratio_[-1]:.1%} of the variance"
            )
    try:
        ld = LDA(shrinkage=0.05, scale=True).fit(X_train, y_train)
        m = make_model().fit(ld.transform(X_train), y_train)
        out["lda"] = float(roc_auc(y_test, m.predict_proba(ld.transform(X_test))[:, 1]))
        if verbose:
            print(f"  {'LDA':<20} {ld.n_components_:>3} features | test ROC-AUC {out['lda']:.4f}")
    except Exception as e:
        if verbose:
            print(f"  LDA skipped: {e}")
    return out
