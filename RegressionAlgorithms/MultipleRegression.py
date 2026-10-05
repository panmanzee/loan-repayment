"""
Multiple Linear Regression (from scratch) - Regression task (target: int.rate)

Same model as LinearRegression.py but X has many feature columns. Adds optional Ridge (L2) regularisation
(l2=0 gives plain least squares) and a standardised-coefficient importance measure.
Scale X (StandardScalerScratch) before using fit_gradient_descent; fit_normal_equation needs no scaling.
Run/import from this folder so `LinearRegression` resolves.
"""
import numpy as np

from LinearRegression import LinearRegressionScratch


class MultipleRegressionScratch(LinearRegressionScratch):
    """
    y_hat = theta_0 + theta_1*x_1 + ... + theta_n*x_n
    Ridge normal equation : theta = (X^T X + l2 * I)^-1 X^T y   (the bias is not penalised)
    Ridge gradient descent: theta -= lr * ( X^T (X theta - y) / m + l2/m * theta )   (bias not penalised)
    """

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
            self.loss_history.append(np.mean(error ** 2))
        return self

    def standardized_importance(self, X):
        """|theta_j| * std(x_j): the change in predicted rate for a 1-SD move in feature j."""
        return np.abs(self.coef_) * X.std(axis=0)
