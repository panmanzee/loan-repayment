"""
Shared data handling (used by every task): load the CSV, build features, split, scale.
Everything except pandas' CSV reading / one-hot encoding is written from scratch.
"""
import os

import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DataSet", "loan_data.csv")


def load_dataset(path=DATA_PATH):
    return pd.read_csv(path)


def build_engineered_frame(df):
    """One-hot `purpose` + engineered features (shared by regression and classification)."""
    out = pd.concat([df, pd.get_dummies(df["purpose"], prefix="purpose", drop_first=True)], axis=1)
    out = out.drop("purpose", axis=1)
    out["installment_to_income"] = out["installment"] / np.exp(out["log.annual.inc"])
    out["high_revol_util"] = (out["revol.util"] > 80).astype(int)
    out["bad_history_flag"] = ((out["delinq.2yrs"] > 0) | (out["pub.rec"] > 0)).astype(int)
    out["years_with_cr_line"] = out["days.with.cr.line"] / 365
    return out.drop("days.with.cr.line", axis=1)


def build_regression_data(df):
    """
    Target = int.rate. `not.fully.paid` is dropped: it is a FUTURE outcome (known only after the
    rate is set), so using it as a feature would be data leakage.
    Returns X (float array), y (float array), feature_cols (list of names).
    """
    frame = build_engineered_frame(df)
    feature_cols = [c for c in frame.columns if c not in ("int.rate", "not.fully.paid")]
    return frame[feature_cols].values.astype(float), frame["int.rate"].values.astype(float), feature_cols


class StandardScalerScratch:
    """z = (x - mean) / std, with mean/std learned from the TRAINING data only."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def train_test_split_scratch(X, y, test_size=0.2, seed=42):
    """Shuffle once with a fixed seed, then cut off the last `test_size` fraction as the test set."""
    idx = np.random.RandomState(seed).permutation(len(y))
    n_test = int(round(test_size * len(y)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def kfold_indices(n_samples, n_splits=5, seed=42):
    """Yield (train_idx, test_idx) for shuffled k-fold cross-validation."""
    idx = np.random.RandomState(seed).permutation(n_samples)
    folds = np.array_split(idx, n_splits)
    for k in range(n_splits):
        train = np.concatenate([folds[j] for j in range(n_splits) if j != k])
        yield train, folds[k]
