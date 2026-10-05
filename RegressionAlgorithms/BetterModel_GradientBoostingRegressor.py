"""
"Better / extracurricular" model: Gradient Boosting Regressor - Regression task (target: int.rate)

Everything is written from scratch with NumPy: the regression tree (weak learner) AND the boosting loop.

INTERNAL MECHANISM
------------------
A single straight line (or polynomial) applies one fixed formula to every borrower. Boosting instead builds
a model as a SUM of many small decision trees, each one correcting what the previous ones still get wrong.

  F_0(x) = mean(y)                                        start: predict the average rate for everyone
  for m = 1 .. M:
      r_i  = y_i - F_{m-1}(x_i)                            residual = how wrong the ensemble still is
      pick a random `subsample` fraction of the rows       stochastic gradient boosting (Friedman, 1999)
      h_m  = small regression tree fitted to (x_i, r_i)    weak learner: predicts the residual
      F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)         take a small step towards fixing the error

For squared-error loss L = 1/2 (y - F)^2, the negative gradient dL/dF is exactly the residual y - F, so
"fit a tree to the residuals" IS gradient descent in function space - hence "gradient" boosting.

The weak learner is a CART regression tree. At each node it tries every feature and every threshold and keeps
the split that most reduces the sum of squared errors (SSE):
      gain = SSE(parent) - SSE(left) - SSE(right)
A leaf predicts the mean residual of its rows. Because the tree can ask "fico < 700 AND revol.util > 60?",
it learns feature interactions and non-linear effects automatically.

Hyper-parameters: n_estimators (M), learning_rate (step size), max_depth (weak-learner size), subsample
(row fraction per tree: adds randomness, reduces over-fitting), min_samples_leaf.
"""
import numpy as np


class RegressionTreeScratch:
    """CART regression tree (squared-error criterion), stored as flat arrays for fast prediction."""

    def __init__(self, max_depth=3, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.feature, self.threshold, self.left, self.right, self.value = [], [], [], [], []
        self.gain_per_feature_ = np.zeros(self.n_features_)  # total SSE reduction credited to each feature
        self._build(X, y, depth=0)
        self.feature = np.array(self.feature)
        self.threshold = np.array(self.threshold)
        self.left = np.array(self.left)
        self.right = np.array(self.right)
        self.value = np.array(self.value)
        return self

    def _new_node(self, value):
        self.feature.append(-1)       # -1 = leaf
        self.threshold.append(0.0)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(value)
        return len(self.value) - 1

    def _best_split(self, X, y):
        """Return (feature, threshold, gain) of the best split, or (None, None, 0) if none is allowed."""
        n = len(y)
        total = y.sum()
        parent_term = total ** 2 / n                  # SSE(parent) = sum(y^2) - total^2/n
        left_n = np.arange(1, n)
        right_n = n - left_n
        size_ok = (left_n >= self.min_samples_leaf) & (right_n >= self.min_samples_leaf)
        best_feat, best_thr, best_gain = None, None, 0.0
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j], kind="stable")
            xs, ys = X[order, j], y[order]
            left_sum = np.cumsum(ys)[:-1]
            right_sum = total - left_sum
            # maximising left_sum^2/nL + right_sum^2/nR  <=>  minimising SSE(left) + SSE(right)
            score = left_sum ** 2 / left_n + right_sum ** 2 / right_n
            allowed = size_ok & (xs[1:] != xs[:-1])   # can only cut between two different values
            if not allowed.any():
                continue
            score = np.where(allowed, score, -np.inf)
            k = int(np.argmax(score))
            gain = score[k] - parent_term
            if gain > best_gain:
                best_feat, best_thr, best_gain = j, (xs[k] + xs[k + 1]) / 2.0, gain
        return best_feat, best_thr, best_gain

    def _build(self, X, y, depth):
        node = self._new_node(float(y.mean()))
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return node
        feat, thr, gain = self._best_split(X, y)
        if feat is None:
            return node
        go_left = X[:, feat] <= thr
        self.feature[node] = feat
        self.threshold[node] = thr
        self.gain_per_feature_[feat] += gain
        self.left[node] = self._build(X[go_left], y[go_left], depth + 1)
        self.right[node] = self._build(X[~go_left], y[~go_left], depth + 1)
        return node

    def predict(self, X):
        node = np.zeros(X.shape[0], dtype=int)
        while True:
            feat = self.feature[node]
            internal = feat >= 0
            if not internal.any():
                break
            rows = np.where(internal)[0]
            go_left = X[rows, feat[rows]] <= self.threshold[node[rows]]
            node[rows] = np.where(go_left, self.left[node[rows]], self.right[node[rows]])
        return self.value[node]


class GradientBoostingRegressorScratch:
    """Stochastic Gradient Boosting Regressor (squared-error loss). See the module docstring."""

    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=3, subsample=1.0,
                 min_samples_leaf=5, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []
        self.initial_pred = None
        self.train_loss = []              # training MSE after each boosting round (the loss curve)

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n = len(y)
        n_sub = max(1, int(round(self.subsample * n)))
        self.trees, self.train_loss = [], []
        self.initial_pred = float(np.mean(y))
        F = np.full(n, self.initial_pred)
        for _ in range(self.n_estimators):
            residuals = y - F
            idx = rng.choice(n, size=n_sub, replace=False) if self.subsample < 1.0 else np.arange(n)
            tree = RegressionTreeScratch(self.max_depth, self.min_samples_leaf).fit(X[idx], residuals[idx])
            self.trees.append(tree)
            F = F + self.learning_rate * tree.predict(X)
            self.train_loss.append(float(np.mean((y - F) ** 2)))
        return self

    def predict(self, X):
        F = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            F = F + self.learning_rate * tree.predict(X)
        return F

    def staged_predict(self, X):
        """Yield the prediction after 0, 1, 2, ... trees (used for the performance curve)."""
        F = np.full(X.shape[0], self.initial_pred)
        yield F.copy()
        for tree in self.trees:
            F = F + self.learning_rate * tree.predict(X)
            yield F.copy()

    @property
    def feature_importances_(self):
        """Share of the total SSE reduction credited to each feature (sums to 1)."""
        total = np.sum([t.gain_per_feature_ for t in self.trees], axis=0)
        return total / total.sum()
