"""ค่าตั้งของทุกโมเดล ใช้ใน notebook เปรียบเทียบ (CV และ batch runs)"""

import numpy as np

from BetterModel_GradientBoostingRegressor import GradientBoostingRegressorScratch
from BlendEnsemble import BlendEnsembleScratch
from data import StandardScalerScratch
from DecisionTree import DecisionTreeClassifierScratch
from evaluation_classification import BETTER
from ExplainableBoosting import ExplainableBoostingScratch
from GradientBoosting import GradientBoostingClassifierScratch
from KNN import KNNClassifierScratch
from LinearRegression import LinearRegressionScratch
from Logisticregression import LogisticRegressionScratch
from MLP import MLPClassifierScratch
from MultipleRegression import MultipleRegressionScratch
from NaiveBayes import NaiveBayesScratch
from Perceptron import PerceptronScratch, SingleLayerPerceptronScratch
from PolynomialRegression import PolynomialRegressionScratch
from RandomForest import RandomForestClassifierScratch
from Stackingclassifier import StackingClassifierScratch, default_base_models
from SVC import SVCScratch
from XGB import XGBClassifierScratch

SEED = 42


class RegressionPipeline:
    def __init__(self, model, columns=None, scale=True):
        self.model = model
        self.columns = columns
        self.scale = scale

    def _select(self, X):
        return X if self.idx_ is None else X[:, self.idx_]

    def fit(self, X, y, feature_cols):
        self.feature_cols_ = list(feature_cols)
        self.idx_ = None if self.columns is None else [self.feature_cols_.index(c) for c in self.columns]
        Xs = self._select(X)
        self.scaler_ = StandardScalerScratch().fit(Xs) if self.scale else None
        Xs = self.scaler_.transform(Xs) if self.scale else Xs
        if hasattr(self.model, "fit_gradient_descent"):
            self.model.fit_gradient_descent(Xs, y)
        else:
            self.model.fit(Xs, y)
        return self

    def transform(self, X):
        Xs = self._select(X)
        return self.scaler_.transform(Xs) if self.scale else Xs

    def predict(self, X):
        return self.model.predict(self.transform(X))


POLY_FEATURES = ["fico", "revol.util", "dti", "inq.last.6mths", "credit.policy"]


def regression_models(gbr=None):
    return {
        "1. Simple Linear (FICO)": lambda: RegressionPipeline(
            LinearRegressionScratch(learning_rate=0.1, n_iterations=2000), columns=["fico"]
        ),
        "2. Multiple Linear": lambda: RegressionPipeline(
            MultipleRegressionScratch(learning_rate=0.1, n_iterations=2000)
        ),
        "3. Polynomial (deg 2)": lambda: RegressionPipeline(
            PolynomialRegressionScratch(degree=2, learning_rate=0.05, n_iterations=3000), columns=POLY_FEATURES
        ),
        "4. Gradient Boosting (better)": lambda: RegressionPipeline(
            GradientBoostingRegressorScratch(
                gbr["n_estimators"], gbr["learning_rate"], gbr["max_depth"], subsample=0.8, random_state=SEED
            ),
            scale=False,
        ),
    }


def classification_models(ebm_weight=0.2, n_mlps=5):
    return {
        "Logistic Regression": (
            lambda: LogisticRegressionScratch(learning_rate=0.5, n_iterations=500, reg_lambda=1.0),
            True,
        ),
        "Decision Tree": (
            lambda: DecisionTreeClassifierScratch(max_depth=6, min_samples_leaf=20, random_state=SEED),
            False,
        ),
        "Random Forest": (
            lambda: RandomForestClassifierScratch(
                n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=SEED
            ),
            False,
        ),
        "Gradient Boosting": (
            lambda: GradientBoostingClassifierScratch(
                n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED
            ),
            True,
        ),
        "XGBoost": (
            lambda: XGBClassifierScratch(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
            ),
            True,
        ),
        "SVC (linear)": (
            lambda: SVCScratch(
                C=1.0,
                kernel="linear",
                solver="primal",
                n_iterations=1500,
                scale_pos_weight="balanced",
                random_state=SEED,
            ),
            True,
        ),
        "SVC (rbf)": (
            lambda: SVCScratch(C=1.0, kernel="rbf", n_iterations=800, scale_pos_weight="balanced", random_state=SEED),
            True,
        ),
        "k-NN": (
            lambda: KNNClassifierScratch(
                n_neighbors=25, weights="distance", tune_k="auto", scale_pos_weight="balanced", random_state=SEED
            ),
            True,
        ),
        "Naive Bayes": (lambda: NaiveBayesScratch(kind="categorical", n_bins=10), False),
        "Perceptron (averaged)": (
            lambda: PerceptronScratch(variant="averaged", learning_rate=0.1, n_epochs=40, random_state=SEED),
            True,
        ),
        "Single-Layer Perceptron": (
            lambda: SingleLayerPerceptronScratch(
                learning_rate=0.05, n_epochs=300, l2=1e-3, lr_decay=0.3, random_state=SEED
            ),
            True,
        ),
        "MLP": (
            lambda: MLPClassifierScratch(
                hidden_layers=(64, 32), l2=1e-2, dropout=0.2, learning_rate=1e-3, n_epochs=200, random_state=SEED
            ),
            True,
        ),
        "EBM (additive)": (
            lambda: ExplainableBoostingScratch(
                n_rounds=800,
                learning_rate=0.02,
                max_bins=16,
                max_leaves=2,
                min_samples_leaf=60,
                reg_lambda=20.0,
                n_outer_bags=8,
                n_interactions=0,
                linear_base=True,
                random_state=SEED,
            ),
            True,
        ),
        "Stacking (5 bases)": (
            lambda: StackingClassifierScratch(default_base_models(SEED), n_folds=5, random_state=SEED),
            True,
        ),
        BETTER: (lambda: BlendEnsembleScratch(n_mlps=n_mlps, ebm_weight=ebm_weight, random_state=SEED), True),
    }


def _plain(v):
    if callable(v):
        return False
    if isinstance(v, (list, tuple)):
        return all(_plain(x) for x in v)
    if isinstance(v, dict):
        return all(_plain(x) for x in v.values())
    return isinstance(v, (int, float, str, bool, type(None), np.integer, np.floating))


def settings_of(model):
    out = {k: v for k, v in vars(model).items() if not k.endswith("_") and _plain(v)}
    inner = getattr(model, "model", None)
    if inner is not None and not callable(inner):
        out.update({f"model.{k}": v for k, v in settings_of(inner).items()})
    for k, v in vars(model).items():
        if k.startswith("_") and hasattr(v, "__dict__") and not callable(v):
            out.update({f"{k}.{kk}": vv for kk, vv in settings_of(v).items()})
    return out


def check_settings(name, model, **zoo_kwargs):
    zoo = (
        regression_models(**zoo_kwargs) if name[:2] in ("1.", "2.", "3.", "4.") else classification_models(**zoo_kwargs)
    )
    ref = zoo[name] if name[:2] in ("1.", "2.", "3.", "4.") else zoo[name][0]
    mine, theirs = settings_of(model), settings_of(ref())
    if mine != theirs:
        diff = {k: (mine.get(k), theirs.get(k)) for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k)}
        raise AssertionError(f"{name}: settings differ from Common/model_zoo.py: {diff}")
    print(f"settings match Common/model_zoo.py ({len(mine)} values checked)")
