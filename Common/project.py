"""
Project-wide paths and the fixed data splits used by every notebook.

Every notebook starts with:
    from project import *            (after adding Common/ to sys.path)
and gets the same data, the same split (seed 42) and the folders where its outputs go:
    Images/<Task>/<Notebook name>/        charts (PNG)
    Results/<Task>/<Notebook name>/       tables (CSV) + predictions for the comparison notebook
    TrainedModels/<Task>/<Notebook name>.joblib
"""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _folder in ("Common", "RegressionAlgorithms", "ClassificationAlgorithms"):
    _path = os.path.join(ROOT, _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data import build_regression_data, load_dataset, train_test_split_scratch  # noqa: E402
from data_classification import build_classification_data, train_val_test_split_scratch  # noqa: E402

SEED = 42


class Outputs:
    """Where one notebook writes its charts, tables and trained model."""

    def __init__(self, task, name):
        self.task, self.name = task, name
        self.images = os.path.join(ROOT, "Images", task, name)
        self.results = os.path.join(ROOT, "Results", task, name)
        self.models = os.path.join(ROOT, "TrainedModels", task)
        for folder in (self.images, self.results, self.models):
            os.makedirs(folder, exist_ok=True)

    def image(self, filename):
        return os.path.join(self.images, filename)

    def table(self, filename):
        return os.path.join(self.results, filename)

    def model(self):
        return os.path.join(self.models, f"{self.name}.joblib")

    def save_json(self, filename, obj):
        with open(self.table(filename), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=float)


def load_json(task, name, filename):
    with open(os.path.join(ROOT, "Results", task, name, filename), encoding="utf-8") as f:
        return json.load(f)


def regression_data(seed=SEED):
    """Task A: target int.rate, 80/20 train/test split (the scaler is fitted per model, on train only)."""
    df = load_dataset()
    X, y, feature_cols = build_regression_data(df)
    X_train, X_test, y_train, y_test = train_test_split_scratch(X, y, test_size=0.2, seed=seed)
    return SimpleNamespace(
        df=df, X=X, y=y, feature_cols=feature_cols, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )


def classification_data(seed=SEED):
    """Task B: target not.fully.paid (16% positives), 60/20/20 train/validation/test split.

    train      -> fit the model
    validation -> early stopping, hyper-parameters and the flagging threshold
    test       -> touched once, for the reported numbers
    """
    df = load_dataset()
    X, y, feature_cols = build_classification_data(df)
    X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test_split_scratch(X, y, seed=seed)
    return SimpleNamespace(
        df=df, X=X, y=y, feature_cols=feature_cols, X_tr=X_tr, X_va=X_va, X_te=X_te, y_tr=y_tr, y_va=y_va, y_te=y_te
    )


def imbalance_weight(y):
    """Project convention for the 16% positive class: positives weigh sqrt(n_negative / n_positive)."""
    y = np.asarray(y)
    return np.where(y == 1, np.sqrt((len(y) - y.sum()) / y.sum()), 1.0)
