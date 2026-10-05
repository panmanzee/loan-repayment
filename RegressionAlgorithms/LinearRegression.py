"""
Linear Regression (from scratch) - Regression task (target: int.rate)

Simple regression = 1 feature (e.g. fico). Fitting methods: Normal Equation and Gradient Descent.
Metrics (loss, R2, RMSE) live in Common/metrics.py.
"""
import numpy as np


class LinearRegressionScratch:
    """
    Linear Regression from scratch.
    Supports Simple (1 feature) and Multiple (all features).
    Two fitting methods: Normal Equation and Gradient Descent.
    """
    def __init__(self, learning_rate=0.1, n_iterations=2000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.theta = None
        self.loss_history = []

    def _add_bias(self, X):
        return np.c_[np.ones(X.shape[0]), X]

    def fit_normal_equation(self, X, y):
        """
        Exact closed-form solution: theta = (X^T X)^{-1} X^T y
        Uses pinv for numerical stability. No scaling required.
        """
        Xb = self._add_bias(X)
        self.theta = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y
        return self

    def fit_gradient_descent(self, X, y):
        """
        Iterative gradient descent. Requires scaled X.
        Update rule: theta -= (lr/m) * X^T * (X*theta - y)
        """
        Xb = self._add_bias(X)
        m = Xb.shape[0]
        self.theta = np.zeros(Xb.shape[1])
        self.loss_history = []
        for _ in range(self.n_iter):
            error = Xb @ self.theta - y
            self.theta -= (self.lr / m) * (Xb.T @ error)
            self.loss_history.append(np.mean(error ** 2))
        return self

    def predict(self, X):
        return self._add_bias(X) @ self.theta

    @property
    def intercept_(self):
        return self.theta[0]

    @property
    def coef_(self):
        return self.theta[1:]
