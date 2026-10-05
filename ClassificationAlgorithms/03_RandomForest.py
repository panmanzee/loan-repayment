"""
Random Forest
Classification task (target: not.fully.paid, 1 = defaulted)

Status: scikit-learn version only. The from-scratch version required by the course spec is NOT implemented yet.
Decision threshold: 0.30 on predict_proba (favours recall on the minority default class).
"""
from sklearn.ensemble import RandomForestClassifier

THRESHOLD = 0.30


def build_model():
    return RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
