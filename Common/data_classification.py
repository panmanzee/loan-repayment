import os

import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DataSet", "loan_data.csv")


def load_dataset(path=DATA_PATH):
    return pd.read_csv(path)


def build_engineered_frame(df):
    out = df.copy()
    out["annual_inc"] = np.exp(out["log.annual.inc"])
    out["installment_to_inc"] = (out["installment"] * 12) / (out["annual_inc"] + 1)
    out["risk_score"] = (
        (out["inq.last.6mths"] >= 3).astype(int)
        + (out["revol.util"] >= 70).astype(int)
        + (out["pub.rec"] > 0).astype(int)
        + (out["fico"] < 690).astype(int)
        + (out["installment_to_inc"] >= 0.10).astype(int)
        + (out["purpose"] == "small_business").astype(int)
    )
    out["revol_stress"] = out["revol.bal"] * (out["revol.util"] / 100)
    out["fico_int_rate_ratio"] = out["fico"] / (out["int.rate"] * 100)
    out["policy_inquiry_risk"] = (1 - out["credit.policy"]) * 3 + out["inq.last.6mths"].clip(upper=6)

    out["util_fico_mismatch"] = out["revol.util"] / (out["fico"] + 1e-6)
    out["policy_fico_mismatch"] = ((out["credit.policy"] == 0) & (out["fico"] > out["fico"].median())).astype(int)

    out["revol_bal_to_inc"] = out["revol.bal"] / (out["annual_inc"] + 1)
    out["inq_per_credit_age"] = out["inq.last.6mths"] / ((out["days.with.cr.line"] / 365) + 1)
    out = pd.concat([out,pd.get_dummies(out["purpose"], prefix="purpose", drop_first=True),],axis=1,)
    out = out.drop(columns=["purpose","annual_inc"])
    return out

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

def build_classification_data(df):

    frame = build_engineered_frame(df)

    feature_cols = [c for c in frame.columns if c != "not.fully.paid"]

    X = frame[feature_cols].values.astype(float)
    y = frame["not.fully.paid"].values.astype(int) 

    return X, y, feature_cols

def train_val_test_split_scratch(X, y, val_size=0.2, test_size=0.2, seed=42):
    n = len(y)
    idx = np.random.RandomState(seed).permutation(n)

    n_test = int(round(test_size * n))
    n_val = int(round(val_size * n))

    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    return (
        X[train_idx],
        X[val_idx],
        X[test_idx],
        y[train_idx],
        y[val_idx],
        y[test_idx],
    )

def kfold_indices(n_samples, n_splits=5, seed=42):
    """Yield (train_idx, test_idx) for shuffled k-fold cross-validation."""
    idx = np.random.RandomState(seed).permutation(n_samples)
    folds = np.array_split(idx, n_splits)
    for k in range(n_splits):
        train = np.concatenate([folds[j] for j in range(n_splits) if j != k])
        yield train, folds[k]
