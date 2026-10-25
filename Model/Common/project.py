"""ค่าที่ทุก notebook ใช้ร่วมกัน: seed, ข้อมูล, โฟลเดอร์ Results"""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SRC)
for _folder in ("Common", "Regression", "Classification"):
    _path = os.path.join(SRC, _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data import build_regression_data, load_dataset, train_test_split_scratch
from data_classification import build_classification_data, train_val_test_split_scratch

SEED = 42


class Outputs:
    def __init__(self, task, name):
        self.task, self.name = task, name
        self.results = os.path.join(ROOT, "Results", task, name)
        self.images = self.models = self.results
        os.makedirs(self.results, exist_ok=True)

    def image(self, filename):
        return os.path.join(self.images, filename)

    def table(self, filename):
        return os.path.join(self.results, filename)

    def model(self):
        return os.path.join(self.models, "model.joblib")

    def save_json(self, filename, obj):
        with open(self.table(filename), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=float)


def load_json(task, name, filename):
    with open(os.path.join(ROOT, "Results", task, name, filename), encoding="utf-8") as f:
        return json.load(f)


def regression_data(seed=SEED):
    df = load_dataset()
    X, y, feature_cols = build_regression_data(df)
    X_train, X_test, y_train, y_test = train_test_split_scratch(X, y, test_size=0.2, seed=seed)
    return SimpleNamespace(
        df=df, X=X, y=y, feature_cols=feature_cols, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )


def classification_data(seed=SEED):
    df = load_dataset()
    X, y, feature_cols = build_classification_data(df)
    X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test_split_scratch(X, y, seed=seed)
    return SimpleNamespace(
        df=df, X=X, y=y, feature_cols=feature_cols, X_tr=X_tr, X_va=X_va, X_te=X_te, y_tr=y_tr, y_va=y_va, y_te=y_te
    )


def imbalance_weight(y):
    y = np.asarray(y)
    return np.where(y == 1, np.sqrt((len(y) - y.sum()) / y.sum()), 1.0)
