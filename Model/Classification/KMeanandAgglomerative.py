"""k-Means และ Agglomerative Clustering"""

import time

import numpy as np


def _as_float(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or inf - clustering compares distances, it cannot skip them")
    return X


def _default_names(n, prefix="feat"):
    return [f"{prefix}{i + 1}" for i in range(n)]


def _pairwise_euclidean(A, B, block=512):
    out = np.empty((len(A), len(B)), dtype=float)
    bb = np.sum(B * B, axis=1).reshape(1, -1)
    for start in range(0, len(A), block):
        chunk = A[start : start + block]
        aa = np.sum(chunk * chunk, axis=1).reshape(-1, 1)
        out[start : start + len(chunk)] = np.sqrt(np.maximum(aa - 2.0 * (chunk @ B.T) + bb, 0.0))
    return out


def _pairwise(A, B, metric="euclidean", block=512):
    if metric == "euclidean":
        return _pairwise_euclidean(A, B, block)
    if metric == "manhattan":
        out = np.empty((len(A), len(B)), dtype=float)
        for start in range(0, len(A), block):
            chunk = A[start : start + block]
            out[start : start + len(chunk)] = np.abs(chunk[:, None, :] - B[None, :, :]).sum(axis=2)
        return out
    if metric == "cosine":
        na = np.linalg.norm(A, axis=1, keepdims=True)
        nb = np.linalg.norm(B, axis=1, keepdims=True).T
        return 1.0 - (A @ B.T) / np.maximum(na * nb, 1e-12)
    raise ValueError('metric must be "euclidean", "manhattan" or "cosine"')


class _ClusterBase:
    def _fit_scaler(self, X):
        self.mean_ = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        self.scale_ = sd

    def _z(self, X):
        Z = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        if self.scale and self.clip is not None:
            Z = np.clip(Z, -float(self.clip), float(self.clip))
        return Z

    def _prepare(self, X, feature_names=None):
        X = _as_float(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = list(feature_names or self.feature_names or _default_names(self.n_features_in_))
        if len(self.feature_names_) != self.n_features_in_:
            raise ValueError("feature_names has the wrong length")
        if self.scale:
            self._fit_scaler(X)
        else:
            self.mean_ = np.zeros(self.n_features_in_)
            self.scale_ = np.ones(self.n_features_in_)
        return X, self._z(X)

    def _check_fitted(self):
        if not getattr(self, "fitted_", False):
            raise RuntimeError(f"{type(self).__name__} is not fitted yet - call fit(X) first")

    def cluster_profile(self, X, top=5):
        self._check_fitted()
        Z = self._z(X)
        overall = Z.mean(axis=0)
        rows = []
        for c in range(self.n_clusters_):
            m = self.labels_ == c
            if m.sum() == 0:
                rows.append({"cluster": c, "n": 0, "share": 0.0, "top_features": []})
                continue
            delta = Z[m].mean(axis=0) - overall
            order = np.argsort(-np.abs(delta))[:top]
            rows.append(
                {
                    "cluster": int(c),
                    "n": int(m.sum()),
                    "share": float(m.mean()),
                    "top_features": [(self.feature_names_[j], float(delta[j])) for j in order],
                }
            )
        return rows

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


def silhouette_score(X, labels, metric="euclidean", max_samples=2000, random_state=42):
    X = _as_float(X)
    labels = np.asarray(labels).astype(int).ravel()
    if len(labels) != len(X):
        raise ValueError("X and labels have different lengths")
    uniq = np.unique(labels)
    if len(uniq) < 2:
        raise ValueError("the silhouette needs at least two clusters")
    S = np.array([np.mean(labels == c) for c in uniq])
    if (S == 1).any():
        raise ValueError("every row is in its own cluster - the silhouette is not defined")
    if len(X) > max_samples:
        rng = np.random.RandomState(random_state)
        take = rng.choice(len(X), int(max_samples), replace=False)
        X, labels = X[take], labels[take]
    D = _pairwise(X, X, metric)
    out = np.zeros(len(X))
    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False
        if same.sum() == 0:
            out[i] = 0.0
            continue
        a = D[i, same].mean()
        b = np.inf
        for c in np.unique(labels):
            if c == labels[i]:
                continue
            other = labels == c
            if other.any():
                b = min(b, D[i, other].mean())
        out[i] = 0.0 if max(a, b) <= 0 else (b - a) / max(a, b)
    return float(out.mean())


def inertia_of(X, labels, centers=None):
    X = _as_float(X)
    labels = np.asarray(labels).astype(int).ravel()
    total = 0.0
    for c in np.unique(labels):
        rows = X[labels == c]
        mu = rows.mean(axis=0) if centers is None else np.asarray(centers)[c]
        total += float(np.sum((rows - mu) ** 2))
    return total


def contingency_table(labels, y):
    labels = np.asarray(labels).astype(int).ravel()
    y = np.asarray(y).astype(int).ravel()
    cl = np.unique(labels)
    yy = np.unique(y)
    return np.array([[int(np.sum((labels == c) * (y == v))) for v in yy] for c in cl]), cl, yy


def adjusted_rand_index(y_true, y_pred):
    a = np.asarray(y_true).astype(int).ravel()
    b = np.asarray(y_pred).astype(int).ravel()
    if len(a) != len(b):
        raise ValueError("the two labellings have different lengths")
    table, _, _ = contingency_table(a, b)
    comb2 = lambda v: v * (v - 1.0) / 2.0
    sum_ij = comb2(table).sum()
    sum_i = comb2(table.sum(axis=1)).sum()
    sum_j = comb2(table.sum(axis=0)).sum()
    n = len(a)
    total = comb2(n)
    expected = sum_i * sum_j / total if total > 0 else 0.0
    maximum = 0.5 * (sum_i + sum_j)
    return float((sum_ij - expected) / (maximum - expected)) if maximum != expected else 1.0


def purity(labels, y):
    table, _, _ = contingency_table(labels, y)
    return float(table.max(axis=1).sum() / max(len(np.asarray(labels).ravel()), 1))


def cluster_default_profile(labels, y, cluster_names=None):
    labels = np.asarray(labels).astype(int).ravel()
    y = np.asarray(y).astype(int).ravel()
    base = float(y.mean())
    rows = []
    for c in np.unique(labels):
        m = labels == c
        rate = float(y[m].mean()) if m.sum() else 0.0
        rows.append(
            {
                "cluster": int(c),
                "n": int(m.sum()),
                "default_rate": rate,
                "lift_vs_overall": (rate / base if base > 0 else float("nan")),
            }
        )
    return rows


def format_default_profile(rows, cluster_names=None):
    out = ["  cluster      n   default rate   lift vs overall"]
    for r in rows:
        name = f"cluster {r['cluster']}" if cluster_names is None else str(cluster_names[r["cluster"]])
        out.append(f"  {name:<11} {r['n']:>6}   {r['default_rate']:>9.2%}   {r['lift_vs_overall']:>10.2f}x")
    return "\n".join(out)


class KMeansScratch(_ClusterBase):
    def __init__(
        self,
        n_clusters=8,
        init="kmeans++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        scale=True,
        clip=5.0,
        block=512,
        random_state=42,
        feature_names=None,
    ):
        if init not in ("kmeans++", "random"):
            raise ValueError('init must be "kmeans++" or "random"')
        if int(n_clusters) < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.scale = bool(scale)
        self.clip = clip
        self.block = int(block)
        self.random_state = random_state
        self.feature_names = feature_names

    def _init_centers(self, Z, k, rng):
        if self.init == "random":
            return Z[rng.choice(len(Z), k, replace=False)].copy()
        centres = [Z[rng.randint(len(Z))]]
        d2 = np.sum((Z - centres[0]) ** 2, axis=1)
        for _ in range(1, k):
            total = float(d2.sum())
            if total <= 0:
                centres.append(Z[rng.randint(len(Z))])
            else:
                centres.append(Z[int(np.searchsorted(np.cumsum(d2 / total), rng.rand()))])
            d2 = np.minimum(d2, np.sum((Z - centres[-1]) ** 2, axis=1))
        return np.array(centres)

    def _assign(self, Z, C):
        best_d = np.empty(len(Z))
        best_i = np.empty(len(Z), dtype=int)
        for start in range(0, len(Z), self.block):
            chunk = Z[start : start + self.block]
            D = _pairwise_euclidean(chunk, C)
            best_i[start : start + len(chunk)] = np.argmin(D, axis=1)
            best_d[start : start + len(chunk)] = np.min(D, axis=1) ** 2
        return best_i, best_d

    def _lloyd(self, Z, C, k):
        hist = []
        n_iter = 0
        for it in range(1, self.max_iter + 1):
            labels, d2 = self._assign(Z, C)
            inertia = float(d2.sum())
            hist.append(inertia)
            newC = np.empty_like(C)
            for c in range(k):
                rows = Z[labels == c]
                if len(rows) == 0:
                    far = int(np.argmax(d2))
                    newC[c] = Z[far]
                    d2[far] = 0.0
                else:
                    newC[c] = rows.mean(axis=0)
            shift = float(np.max(np.linalg.norm(newC - C, axis=1)))
            C = newC
            n_iter = it
            if shift <= self.tol:
                break
        labels, d2 = self._assign(Z, C)
        hist.append(float(d2.sum()))
        return C, labels, float(d2.sum()), n_iter, hist

    def fit(self, X, y=None, feature_names=None, **_ignored):
        X, Z = self._prepare(X, feature_names)
        n = len(Z)
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        if self.n_clusters > n:
            raise ValueError(f"n_clusters={self.n_clusters} is larger than the number of rows ({n})")
        k = self.n_clusters
        t0 = time.time()

        if k == 1:
            self.cluster_centers_ = Z.mean(axis=0, keepdims=True)
            self.labels_ = np.zeros(n, dtype=int)
            self.inertia_ = float(np.sum((Z - self.cluster_centers_) ** 2))
            self.n_iter_, self.history_ = 1, [self.inertia_]
        else:
            rng = np.random.RandomState(self.random_state)
            best = None
            for run in range(max(1, self.n_init)):
                C0 = self._init_centers(Z, k, rng)
                C, labels, inertia, n_iter, hist = self._lloyd(Z, C0, k)
                if best is None or inertia < best[0]:
                    best = (inertia, C, labels, n_iter, hist)
            self.inertia_, self.cluster_centers_, self.labels_, self.n_iter_, self.history_ = best

        self.n_clusters_ = k
        self.X_fit_ = Z.copy()
        self.cluster_centers_raw_ = self.cluster_centers_ * self.scale_ + self.mean_
        self.silhouette_ = None
        self.fitted_ = True
        self.fit_time_ = time.time() - t0
        return self

    def fit_predict(self, X, y=None, feature_names=None, **_ignored):
        return self.fit(X, y, feature_names=feature_names).labels_

    def predict(self, X):
        self._check_fitted()
        return self._assign(self._z(X), self.cluster_centers_)[0]

    def transform(self, X):
        self._check_fitted()
        return _pairwise_euclidean(self._z(X), self.cluster_centers_)

    def get_silhouette(self, X=None, max_samples=2000):
        self._check_fitted()
        self.silhouette_ = silhouette_score(
            self._z(X) if X is not None else self.X_fit_, self.labels_, max_samples=max_samples
        )
        return self.silhouette_

    def summary(self):
        self._check_fitted()
        k = self.n_clusters_
        counts = [int(np.sum(self.labels_ == c)) for c in range(k)]
        return (
            f"k-Means: k={k} | inertia {self.inertia_:.1f} | iterations {self.n_iter_} "
            f"| init={self.init} x{self.n_init} | clip={self.clip}\n"
            f"  cluster sizes: {counts}"
        )

    def fit_k_curve(self, X, k_values=(2, 3, 4, 5, 6, 8, 10), feature_names=None, verbose=False):
        X = _as_float(X)
        out = {"k": [], "inertia": [], "silhouette": []}
        for k in k_values:
            if k > len(X):
                continue
            m = KMeansScratch(
                n_clusters=int(k),
                init=self.init,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
                scale=self.scale,
                clip=self.clip,
                block=self.block,
                random_state=self.random_state,
            ).fit(X, feature_names=feature_names)
            sil = silhouette_score(m._z(X), m.labels_, max_samples=2000, random_state=self.random_state)
            out["k"].append(int(k))
            out["inertia"].append(m.inertia_)
            out["silhouette"].append(sil)
            if verbose:
                print(f"  k={k:>3} | inertia {m.inertia_:10.1f} | silhouette {sil:.4f}")
        return out

    def get_state(self):
        self._check_fitted()
        return {
            "cls": "KMeansScratch",
            "params": dict(
                n_clusters=self.n_clusters,
                init=self.init,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
                scale=self.scale,
                clip=self.clip,
                block=self.block,
                random_state=self.random_state,
            ),
            "arrays": {
                "centers": self.cluster_centers_,
                "centers_raw": self.cluster_centers_raw_,
                "labels": self.labels_,
                "mean": self.mean_,
                "scale": self.scale_,
                "history": self.history_,
            },
            "meta": {
                "n_clusters_": self.n_clusters_,
                "inertia": self.inertia_,
                "feature_names": self.feature_names_,
                "n_features_in_": self.n_features_in_,
            },
        }

    @classmethod
    def from_state(cls, s):
        m = cls(**s["params"])
        a, meta = s["arrays"], s["meta"]
        m.cluster_centers_, m.cluster_centers_raw_ = a["centers"], a["centers_raw"]
        m.labels_, m.mean_, m.scale_, m.history_ = a["labels"], a["mean"], a["scale"], a["history"]
        m.n_clusters_, m.inertia_ = meta["n_clusters_"], meta["inertia"]
        m.feature_names_, m.n_features_in_ = meta["feature_names"], meta["n_features_in_"]
        m.silhouette_ = None
        m.fitted_ = True
        m.fit_time_ = 0.0
        return m


class AgglomerativeScratch(_ClusterBase):
    def __init__(
        self,
        n_clusters=2,
        linkage="ward",
        metric="euclidean",
        scale=True,
        clip=5.0,
        max_samples=3000,
        feature_names=None,
    ):
        if linkage not in ("ward", "single", "complete", "average"):
            raise ValueError('linkage must be "ward", "single", "complete" or "average"')
        if linkage == "ward" and metric != "euclidean":
            raise ValueError("ward linkage is only defined for the euclidean metric")
        if metric not in ("euclidean", "manhattan", "cosine"):
            raise ValueError('metric must be "euclidean", "manhattan" or "cosine"')
        if int(n_clusters) < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        self.n_clusters = int(n_clusters)
        self.linkage = linkage
        self.metric = metric
        self.scale = bool(scale)
        self.clip = clip
        self.max_samples = int(max_samples)
        self.feature_names = feature_names

    def fit(self, X, y=None, feature_names=None, **_ignored):
        X, Z = self._prepare(X, feature_names)
        n = len(Z)
        if n > self.max_samples:
            raise ValueError(
                f"agglomerative clustering needs an n x n distance matrix; n={n} > "
                f"max_samples={self.max_samples}. Sub-sample the rows (or pick random "
                f"rows yourself) - unlike k-Means this algorithm is not built for the "
                f"full 9578-row table."
            )
        if self.n_clusters < 1 or self.n_clusters > n:
            raise ValueError(f"n_clusters must be between 1 and the number of rows ({n})")
        t0 = time.time()
        self.X_fit_ = Z.copy()
        self.n_leaves_ = n

        D = _pairwise_euclidean(Z, Z) if self.metric == "euclidean" else _pairwise(Z, Z, self.metric)
        np.fill_diagonal(D, np.inf)
        active = np.ones(n, dtype=bool)
        size = np.ones(n, dtype=float)
        node = np.arange(n)
        children, dists, sizes = [], [], []
        n_active = n

        while n_active > 1:
            a, b = self._closest_pair(D)
            d_ab = float(D[a, b])
            na, nb = size[a], size[b]
            new_id = n + len(children)
            children.append((int(node[a]), int(node[b])))
            dists.append(d_ab)
            sizes.append(na + nb)
            if self.linkage == "single":
                new_d = np.minimum(D[a], D[b])
            elif self.linkage == "complete":
                new_d = np.maximum(D[a], D[b])
            elif self.linkage == "average":
                new_d = (na * D[a] + nb * D[b]) / (na + nb)
            else:
                wc = size.copy()
                new_d = np.sqrt(
                    np.maximum(((na + wc) * D[a] ** 2 + (nb + wc) * D[b] ** 2 - wc * d_ab**2) / (na + nb + wc), 0.0)
                )
            new_d[a] = np.inf
            D[a, :] = new_d
            D[:, a] = new_d
            D[a, a] = np.inf
            D[:, b] = np.inf
            D[b, :] = np.inf
            active[b] = False
            size[a] = na + nb
            node[a] = new_id
            n_active -= 1

        self.children_ = children
        self.merge_distances_ = dists
        self.merge_sizes_ = sizes
        self.n_leaves_ = n
        self.labels_ = self.cut_into(self.n_clusters)
        self.n_clusters_ = self.n_clusters
        self.fitted_ = True
        self.fit_time_ = time.time() - t0
        return self

    @staticmethod
    def _closest_pair(D):
        flat = int(np.argmin(D))
        return flat // D.shape[0], flat % D.shape[1]

    def cut_into(self, n_clusters):
        if not hasattr(self, "children_"):
            raise RuntimeError("fit() first")
        n = self.n_leaves_
        if n_clusters < 1 or n_clusters > n:
            raise ValueError(f"n_clusters must be between 1 and {n}")
        parent = np.arange(2 * n - 1)
        merges_needed = n - n_clusters
        for k in range(merges_needed):
            a, b = self.children_[k]
            new_id = n + k
            parent[a] = new_id
            parent[b] = new_id
        labels = np.empty(n, dtype=int)
        roots = {}
        for leaf in range(n):
            r = leaf
            while parent[r] != r:
                r = parent[r]
            roots.setdefault(int(r), len(roots))
            labels[leaf] = roots[int(r)]
        return labels

    def fit_predict(self, X, y=None, feature_names=None, **_ignored):
        return self.fit(X, y, feature_names=feature_names).labels_

    def summary(self):
        self._check_fitted()
        counts = [int(np.sum(self.labels_ == c)) for c in range(self.n_clusters_)]
        last = self.merge_distances_[len(self.children_) - 1] if self.children_ else 0.0
        return (
            f"Agglomerative: linkage={self.linkage} metric={self.metric} | "
            f"{self.n_leaves_} rows -> {self.n_clusters_} clusters | "
            f"last merge distance {last:.3f}\n  cluster sizes: {counts}"
        )

    def get_state(self):
        self._check_fitted()
        return {
            "cls": "AgglomerativeScratch",
            "params": dict(
                n_clusters=self.n_clusters,
                linkage=self.linkage,
                metric=self.metric,
                scale=self.scale,
                clip=self.clip,
                max_samples=self.max_samples,
            ),
            "arrays": {
                "children": self.children_,
                "merge_distances": self.merge_distances_,
                "merge_sizes": self.merge_sizes_,
                "labels": self.labels_,
                "mean": self.mean_,
                "scale": self.scale_,
            },
            "meta": {
                "n_clusters_": self.n_clusters_,
                "n_leaves_": self.n_leaves_,
                "feature_names": self.feature_names_,
                "n_features_in_": self.n_features_in_,
            },
        }

    @classmethod
    def from_state(cls, s):
        m = cls(**s["params"])
        a, meta = s["arrays"], s["meta"]
        m.children_, m.merge_distances_, m.merge_sizes_ = a["children"], a["merge_distances"], a["merge_sizes"]
        m.labels_, m.mean_, m.scale_ = a["labels"], a["mean"], a["scale"]
        m.n_clusters_, m.n_leaves_ = meta["n_clusters_"], meta["n_leaves_"]
        m.feature_names_, m.n_features_in_ = meta["feature_names"], meta["n_features_in_"]
        m.fitted_ = True
        m.fit_time_ = 0.0
        return m
