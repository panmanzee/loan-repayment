"""
Shared evaluation metrics, written from scratch.
Regression: loss (MSE), RMSE, MAE, R2.  Also model-agnostic permutation feature importance.
"""
import numpy as np


def mse(y_true, y_pred):
    """Mean squared error = the regression LOSS."""
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred):
    """R2 = 1 - SS_residual / SS_total (0 = no better than predicting the mean)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def permutation_importance(predict_fn, X, y, n_repeats=3, seed=42):
    """
    Importance of feature j = how much the loss (MSE) rises when column j is shuffled.
    Works for any model; larger = the model relies on that feature more.
    """
    rng = np.random.RandomState(seed)
    base = mse(y, predict_fn(X))
    importances = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        increases = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            increases.append(mse(y, predict_fn(Xp)) - base)
        importances[j] = np.mean(increases)
    return importances
