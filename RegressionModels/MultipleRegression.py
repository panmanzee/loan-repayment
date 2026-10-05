"""
Multiple Linear Regression (from scratch) - Regression task (target: int.rate)

Same math as LinearRegression.py, but X has many feature columns.
Scale X (StandardScaler) before using fit_gradient_descent; fit_normal_equation needs no scaling.
Run/import from this folder so `LinearRegression` resolves.
"""
from LinearRegression import LinearRegressionScratch


class MultipleRegressionScratch(LinearRegressionScratch):
    """Multiple Linear Regression: LinearRegressionScratch fitted on several features."""
