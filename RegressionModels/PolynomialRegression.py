"""
Polynomial Regression (from scratch) - Regression task (target: int.rate)

Expands features (original + squared + pairwise products), then fits linear regression by gradient descent.
Run/import from this folder so `LinearRegression` resolves.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler  # only used to scale expanded features

from LinearRegression import LinearRegressionScratch


class PolynomialRegressionScratch:
    """
    Polynomial Regression from scratch (degree=2).
    Manually expands features: original + squared + pairwise interactions.
    Then fits via gradient descent.
    """
    def __init__(self, degree=2, learning_rate=0.05, n_iterations=3000):
        self.degree  = degree
        self._lr_model = LinearRegressionScratch(
            learning_rate=learning_rate, n_iterations=n_iterations)
        self._scaler = StandardScaler()

    def _expand(self, X):
        parts = [X]
        n = X.shape[1]
        if self.degree >= 2:
            parts.append(X ** 2)
            for i in range(n):
                for j in range(i + 1, n):
                    parts.append((X[:, i] * X[:, j]).reshape(-1, 1))
        if self.degree >= 3:
            parts.append(X ** 3)
        return np.hstack(parts)

    def fit(self, X, y):
        Xp = self._expand(X)
        Xp_sc = self._scaler.fit_transform(Xp)
        self._lr_model.fit_gradient_descent(Xp_sc, y)
        return self

    def predict(self, X):
        Xp = self._expand(X)
        Xp_sc = self._scaler.transform(Xp)
        return self._lr_model.predict(Xp_sc)

    @property
    def loss_history(self):
        return self._lr_model.loss_history
