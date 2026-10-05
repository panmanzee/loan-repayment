"""
Dimensionality reduction (e.g. PCA) for Random Forest / SVM
Classification task (target: not.fully.paid, 1 = defaulted)

Status: NOT IMPLEMENTED YET. The course spec requires a from-scratch version
(a scikit-learn version may be added alongside it only to verify results).
"""

import numpy as np


class PCAScratch:
    def fit(self, X, y):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError
