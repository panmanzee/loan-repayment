"""
Ensemble learning: Stacking classifiers / Gradient Boosting / XGBoost
Classification task (target: not.fully.paid, 1 = defaulted)

Status: NOT IMPLEMENTED YET. The course spec requires a from-scratch version
(a scikit-learn version may be added alongside it only to verify results).
"""

import numpy as np


class StackingGBXGBScratch:
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError
