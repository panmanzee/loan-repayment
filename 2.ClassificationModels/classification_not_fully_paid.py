"""
Loan Default Prediction - Classification Task
LendingClub Loan Dataset  (target: not.fully.paid, 1 = defaulted)

Models (sklearn):
  1. Logistic Regression  (class_weight="balanced")
  2. Random Forest        (class_weight="balanced", threshold 0.30)

Run: python classification_not_fully_paid.py
Output: charts saved to charts/cls_*.png,
        final model saved to saved_models/classification_model.joblib
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 120
np.random.seed(42)

print("=" * 60)
print("  Loan Default Prediction - Classification Task")
print("=" * 60)

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # run from this folder
os.makedirs("charts", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# =============================================================================
# 1. DATA PROCESSING & FEATURE ENGINEERING
# =============================================================================
print("\n[1] Loading data & engineering features...")
df = pd.read_csv("../3.Data/loan_data.csv")
df_data = pd.get_dummies(df, columns=["purpose"], drop_first=True)

# Monthly installment relative to real (un-logged) annual income
df_data["installment_to_income"] = df_data["installment"] / np.exp(df_data["log.annual.inc"])

# High FICO but high interest rate = unusual / suspicious.
# Uses int.rate, so it is ONLY valid for classification (never for the int.rate regression).
df_data["fico_rate_gap"] = df_data["fico"] - (df_data["int.rate"] * 1000)

# High-risk flag: revolving credit utilisation above 80%
df_data["high_revol_util"] = (df_data["revol.util"] > 80).astype(int)

# Any bad history (delinquency or public record) merged into one flag
df_data["bad_history_flag"] = ((df_data["delinq.2yrs"] > 0) | (df_data["pub.rec"] > 0)).astype(int)

# Credit line age in years
df_data["years_with_cr_line"] = df_data["days.with.cr.line"] / 365

X = df_data.drop(columns="not.fully.paid")
y = df_data["not.fully.paid"]
feature_cols = list(X.columns)
print(f"    Rows: {len(df_data)} | Features: {len(feature_cols)} | Default rate: {y.mean():.1%}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
# 2. CORRELATION WITH TARGET
# =============================================================================
print("\n[2] Correlation with not.fully.paid...")
target_corr = (
    df_data.astype(float)
    .corr()[["not.fully.paid"]]
    .sort_values(by="not.fully.paid", ascending=False)
)
print(target_corr.to_string())

plt.figure(figsize=(6, 8))
sns.heatmap(target_corr, annot=True, fmt=".3f", cmap="coolwarm", cbar=True)
plt.title("Correlation with not.fully.paid")
plt.savefig("charts/cls_01_target_correlation.png", bbox_inches="tight")
plt.close()
print("    Saved: cls_01_target_correlation.png")

# =============================================================================
# 3. SCALE ONCE (shared by every model)
# =============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def evaluate(name, y_proba, threshold, cmap, chart_path):
    """Print metrics and save a confusion-matrix chart."""
    y_pred = (y_proba >= threshold).astype(int)
    print(f"\n=== {name} (threshold = {threshold}) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC :", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))

    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap=cmap)
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {os.path.basename(chart_path)}")


# =============================================================================
# 4. LOGISTIC REGRESSION
# =============================================================================
print("\n[3] Training Logistic Regression...")
logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
proba_log = logreg.predict_proba(X_test_scaled)[:, 1]
evaluate("Logistic Regression", proba_log, 0.50, "Oranges",
         "charts/cls_02_confusion_logistic.png")

# =============================================================================
# 5. RANDOM FOREST
# =============================================================================
print("\n[4] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf.fit(X_train_scaled, y_train)
proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
# Low threshold (0.30) favours recall on the minority (default) class.
evaluate("Random Forest", proba_rf, 0.30, "Blues",
         "charts/cls_03_confusion_random_forest.png")

# =============================================================================
# 6. ROC CURVE COMPARISON
# =============================================================================
print("\n[5] ROC curve comparison...")
auc_log = roc_auc_score(y_test, proba_log)
auc_rf = roc_auc_score(y_test, proba_rf)
fpr_log, tpr_log, _ = roc_curve(y_test, proba_log)
fpr_rf, tpr_rf, _ = roc_curve(y_test, proba_rf)

plt.figure(figsize=(7, 6))
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC={auc_log:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.savefig("charts/cls_04_roc_comparison.png", bbox_inches="tight")
plt.close()
print("    Saved: cls_04_roc_comparison.png")

# =============================================================================
# 7. SAVE MODEL (only the final classification model = higher ROC-AUC)
# =============================================================================
print("\n[6] Saving final classification model...")
if auc_log >= auc_rf:
    best_name, best_model, best_threshold, best_auc = "Logistic Regression", logreg, 0.50, auc_log
else:
    best_name, best_model, best_threshold, best_auc = "Random Forest", rf, 0.30, auc_rf

classification_bundle = {
    "model": best_model,
    "model_name": best_name,
    "scaler": scaler,
    "feature_cols": feature_cols,
    "threshold": best_threshold,
    "target": "not.fully.paid",
    "test_roc_auc": best_auc,
}
joblib.dump(classification_bundle, "saved_models/classification_model.joblib")
print(f"    Best model: {best_name} (ROC-AUC = {best_auc:.3f})")
print("    Saved: saved_models/classification_model.joblib")

print("\n" + "=" * 60)
print("  DONE. All plots and the model saved.")
print("=" * 60)
