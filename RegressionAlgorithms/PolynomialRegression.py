"""
Polynomial Regression (from scratch) - Regression task (target: int.rate)

Expands features (original + squared + pairwise products), then fits linear regression by gradient descent.
Needs `Common/` and this folder on sys.path (run_regression.py does this).
"""
import numpy as np
from data import StandardScalerScratch  # Common/data.py
from LinearRegression import LinearRegressionScratch


class PolynomialRegressionScratch:
    """
    Polynomial Regression of degree 2, from scratch.

    Idea: a straight line cannot bend, but if we hand the model x^2 as an extra input column it can learn
    y = a*x + b*x^2 + c, which IS a curve. So polynomial regression = linear regression on expanded features.
    With 5 inputs the expansion gives 5 (original) + 5 (squared) + 10 (pairwise products) = 20 columns.
    The expanded columns are standardised and fitted by gradient descent (LinearRegressionScratch).
    """

    def __init__(self, degree=2, learning_rate=0.05, n_iterations=3000):
        if degree != 2:
            raise ValueError("Only degree=2 is implemented")
        self.degree = degree
        self._lr_model = LinearRegressionScratch(learning_rate=learning_rate, n_iterations=n_iterations)
        self._scaler = StandardScalerScratch()

    def _expand(self, X):
        """[x_i] + [x_i^2] + [x_i * x_j for i < j]"""
        n = X.shape[1]
        squares = X ** 2
        products = [(X[:, i] * X[:, j]).reshape(-1, 1) for i in range(n) for j in range(i + 1, n)]
        return np.hstack([X, squares] + products)

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
