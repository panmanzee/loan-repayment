"""
"Better / extracurricular" model: Gradient Boosting Regressor (from scratch) - Regression task (target: int.rate)

Boosting loop and row subsampling are written from scratch; sklearn's DecisionTreeRegressor is the weak learner.
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressorScratch:
    """
    Gradient Boosting Regressor from scratch (Stochastic Gradient Boosting).

    Internal mechanism (MSE loss):
      F_0(x) = mean(y)
      For m = 1..M:
        r_i    = y_i - F_{m-1}(x_i)          <- pseudo-residuals
        (X_sub, r_sub) = random `subsample` fraction of rows  <- stochastic GB (Friedman, 1999)
        h_m    = DecisionTree fitted to (X_sub, r_sub)         <- weak learner
        F_m(x) = F_{m-1}(x) + lr * h_m(x)   <- ensemble update

    The boosting algorithm (incl. row subsampling) is from scratch.
    Uses sklearn DecisionTreeRegressor as the base weak learner.

    subsample < 1.0 trains each tree on a random fraction of rows, which both
    regularizes the ensemble (reduces overfitting/variance) and speeds up fitting
    -- this is what let hyperparameter tuning push R2 from 0.75 to ~0.76 on this
    dataset (see Section 8 tuning search).
    """
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=3, subsample=1.0, random_state=42):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.subsample     = subsample
        self.random_state  = random_state
        self.trees         = []
        self.initial_pred  = None
        self.train_loss    = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n = len(y)
        m_sub = max(1, int(round(self.subsample * n)))
        self.initial_pred = np.mean(y)
        F = np.full(n, self.initial_pred)
        for m in range(self.n_estimators):
            residuals = y - F
            if self.subsample < 1.0:
                idx = rng.choice(n, size=m_sub, replace=False)
                X_fit, r_fit = X[idx], residuals[idx]
            else:
                X_fit, r_fit = X, residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X_fit, r_fit)
            self.trees.append(tree)
            F = F + self.learning_rate * tree.predict(X)
            self.train_loss.append(np.mean((y - F) ** 2))
        return self

    def predict(self, X):
        F = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            F = F + self.learning_rate * tree.predict(X)
        return F

    def staged_predict(self, X):
        F = np.full(X.shape[0], self.initial_pred)
        yield F.copy()
        for tree in self.trees:
            F = F + self.learning_rate * tree.predict(X)
            yield F.copy()
