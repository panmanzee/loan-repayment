"""
Logistic Regression
Classification task (target: not.fully.paid, 1 = defaulted)

Status: scikit-learn version only. The from-scratch version required by the course spec is NOT implemented yet.
Expects StandardScaler-scaled features. Decision threshold: 0.50 on predict_proba.
"""
from sklearn.linear_model import LogisticRegression

THRESHOLD = 0.50


def build_model():
    return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
