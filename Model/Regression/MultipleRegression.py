"""Multiple Linear Regression ใช้ทุก feature"""

import numpy as np

from LinearRegression import LinearRegressionScratch


class MultipleRegressionScratch(LinearRegressionScratch):
    def __init__(self, learning_rate=0.1, n_iterations=2000, l2=0.0):
        super().__init__(learning_rate, n_iterations)
        self.l2 = l2

    def fit_normal_equation(self, X, y):
        Xb = self._add_bias(X)
        penalty = self.l2 * np.eye(Xb.shape[1])
        penalty[0, 0] = 0.0
        self.theta = np.linalg.pinv(Xb.T @ Xb + penalty) @ Xb.T @ y
        return self

    def fit_gradient_descent(self, X, y):
        Xb = self._add_bias(X)
        m = Xb.shape[0]
        self.theta = np.zeros(Xb.shape[1])
        self.loss_history = []
        for _ in range(self.n_iter):
            error = Xb @ self.theta - y
            grad = Xb.T @ error / m
            grad[1:] += (self.l2 / m) * self.theta[1:]
            self.theta -= self.lr * grad
            self.loss_history.append(np.mean(error**2))
        return self

    def standardized_importance(self, X):
        return np.abs(self.coef_) * X.std(axis=0)
