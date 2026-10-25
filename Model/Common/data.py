"""โหลดข้อมูล เตรียม feature งาน Regression และแบ่ง train/test"""

import os

import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "DataSet", "loan_data.csv")


def load_dataset(path=DATA_PATH):
    return pd.read_csv(path)


def build_engineered_frame(df):
    out = pd.concat([df, pd.get_dummies(df["purpose"], prefix="purpose", drop_first=True)], axis=1)
    out = out.drop("purpose", axis=1)
    out["installment_to_income"] = out["installment"] / np.exp(out["log.annual.inc"])
    out["high_revol_util"] = (out["revol.util"] > 80).astype(int)
    out["bad_history_flag"] = ((out["delinq.2yrs"] > 0) | (out["pub.rec"] > 0)).astype(int)
    out["years_with_cr_line"] = out["days.with.cr.line"] / 365
    return out.drop("days.with.cr.line", axis=1)


def build_regression_data(df):
    frame = build_engineered_frame(df)
    feature_cols = [c for c in frame.columns if c not in ("int.rate", "not.fully.paid")]
    return frame[feature_cols].values.astype(float), frame["int.rate"].values.astype(float), feature_cols


class StandardScalerScratch:
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
    idx = np.random.RandomState(seed).permutation(len(y))
    n_test = int(round(test_size * len(y)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def kfold_indices(n_samples, n_splits=5, seed=42):
    idx = np.random.RandomState(seed).permutation(n_samples)
    folds = np.array_split(idx, n_splits)
    for k in range(n_splits):
        train = np.concatenate([folds[j] for j in range(n_splits) if j != k])
        yield train, folds[k]
