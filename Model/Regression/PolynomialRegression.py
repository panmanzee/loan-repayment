"""Polynomial Regression degree 2 (กำลังสอง + ผลคูณคู่ feature)"""

import numpy as np
from data import StandardScalerScratch
from LinearRegression import LinearRegressionScratch


class PolynomialRegressionScratch:
    def __init__(self, degree=2, learning_rate=0.05, n_iterations=3000):
        if degree != 2:
            raise ValueError("Only degree=2 is implemented")
        self.degree = degree
        self._lr_model = LinearRegressionScratch(learning_rate=learning_rate, n_iterations=n_iterations)
        self._scaler = StandardScalerScratch()

    def _expand(self, X):
        n = X.shape[1]
        squares = X**2
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
