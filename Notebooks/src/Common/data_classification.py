"""
Data for the CLASSIFICATION task (target: not.fully.paid): engineered features + the 60/20/20 split.
Loading the CSV, the scaler and k-fold indices are shared with the regression task (Common/data.py).
"""

import numpy as np
import pandas as pd

from data import StandardScalerScratch, kfold_indices, load_dataset  # noqa: F401  (re-exported)


def build_engineered_frame(df):
    """Raw columns + engineered risk features + one-hot `purpose` (all_other is the dropped base level)."""
    out = df.copy()
    out["annual_inc"] = np.exp(out["log.annual.inc"])
    out["installment_to_inc"] = (out["installment"] * 12) / (out["annual_inc"] + 1)  # yearly payment / income
    # count of simple red flags (0-6)
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
    out = pd.concat([out, pd.get_dummies(out["purpose"], prefix="purpose", drop_first=True)], axis=1)
    return out.drop(columns=["purpose", "annual_inc"])


def build_classification_data(df):
    """Target = not.fully.paid. Returns X (float array), y (0/1 array), feature_cols (list of names)."""
    frame = build_engineered_frame(df)
    feature_cols = [c for c in frame.columns if c != "not.fully.paid"]
    return frame[feature_cols].values.astype(float), frame["not.fully.paid"].values.astype(int), feature_cols


def train_val_test_split_scratch(X, y, val_size=0.2, test_size=0.2, seed=42):
    """Shuffle once with a fixed seed: first 20% = test, next 20% = validation, the rest = train."""
    idx = np.random.RandomState(seed).permutation(len(y))
    n_test, n_val = int(round(test_size * len(y))), int(round(val_size * len(y)))
    test_idx, val_idx, train_idx = idx[:n_test], idx[n_test : n_test + n_val], idx[n_test + n_val :]
    return X[train_idx], X[val_idx], X[test_idx], y[train_idx], y[val_idx], y[test_idx]
