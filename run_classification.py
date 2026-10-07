"""
Run the whole CLASSIFICATION task (target: not.fully.paid) -- the twin of run_regression.py.
  load data -> split train/val/test -> train the 11 scratch models -> pick each model's threshold on VALIDATION
  -> test metrics (loss, accuracy, precision/recall/F1, ROC-AUC, PR-AUC, confusion matrix) -> curves
  -> 5-fold cross-validation -> feature importance -> save + load the best model
  -> scratch-vs-scikit-learn checks, EBM shape functions, Naive Bayes densities, explain one borrower (section 10).

Run:  python run_classification.py        (from the project root)
Outputs: printed tables; Results/<topic folders>/ (CSV tables + PNG charts); TrainedModels/classification_model.joblib
"""
import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "Common"))
sys.path.insert(0, os.path.join(ROOT, "ClassificationAlgorithms"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plots  # shared chart style (colours, grid, _save)
from data_classification import (build_classification_data, kfold_indices, load_dataset,
                                 train_val_test_split_scratch)
from DecisionTree import DecisionTreeClassifierScratch
from BlendEnsemble import BlendEnsembleScratch
from ExplainableBoosting import ExplainableBoostingScratch
from GradientBoosting import GradientBoostingClassifierScratch
from Logisticregression import LogisticRegressionScratch
from MLP import MLPClassifierScratch
from NaiveBayes import NaiveBayesScratch
from Perceptron import PerceptronScratch, SingleLayerPerceptronScratch
from metrics_classification import (average_precision, confusion_matrix, evaluate_on_test,
                                    precision_recall_curve_scratch, roc_auc, roc_curve_scratch)
from RandomForest import RandomForestClassifierScratch
from XGB import XGBClassifierScratch

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")
SEED = 42
RESULTS = os.path.join(ROOT, "Results")
FOLDERS = {"data": "01_Data_Analysis", "loss": "02_Loss", "cm": "03_Confusion_Matrix", "acc": "04_Accuracy",
           "prf": "05_Precision_Recall_F1", "roc": "06_ROC_AUC", "perf": "07_Performance_Curve", "other": "09_Others"}
COLORS = {"Logistic Regression": "#2a78d6", "Decision Tree": "#eb6834", "Random Forest": "#1baf7a",
          "Gradient Boosting": "#eda100", "XGBoost": "#8b5cf6",
          "Naive Bayes": "#d4549a", "Perceptron (averaged)": "#0f9bb3", "Single-Layer Perceptron": "#9aa31a",
          "MLP": "#8d6e63", "EBM (additive)": "#111111", "Blend EBM+MLP (better model)": "#d62728"}


def out(topic, filename):
    folder = os.path.join(RESULTS, FOLDERS[topic])
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)


def section(title):
    print("\n" + "=" * 70 + f"\n  {title}\n" + "=" * 70)


def make_models():
    """name -> (factory, uses_validation_set). Fresh model every call so CV folds never share state."""
    return {
        "Logistic Regression": (lambda: LogisticRegressionScratch(learning_rate=0.5, n_iterations=500, reg_lambda=1.0), True),
        "Decision Tree": (lambda: DecisionTreeClassifierScratch(max_depth=6, min_samples_leaf=20, random_state=SEED), False),
        "Random Forest": (lambda: RandomForestClassifierScratch(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=SEED), False),
        "Gradient Boosting": (lambda: GradientBoostingClassifierScratch(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED), True),
        "XGBoost": (lambda: XGBClassifierScratch(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=SEED), True),
        # ---- part 2 (settings chosen on the VALIDATION set only; see docs/PART2_GUIDE.md) ----
        "Naive Bayes": (lambda: NaiveBayesScratch(kind="categorical", n_bins=10), False),
        "Perceptron (averaged)": (lambda: PerceptronScratch(variant="averaged", learning_rate=0.1, n_epochs=40, random_state=SEED), True),
        "Single-Layer Perceptron": (lambda: SingleLayerPerceptronScratch(learning_rate=0.05, n_epochs=300, l2=1e-3, lr_decay=0.3, random_state=SEED), True),
        "MLP": (lambda: MLPClassifierScratch(hidden_layers=(64, 32), l2=1e-2, dropout=0.2, learning_rate=1e-3, n_epochs=200, random_state=SEED), True),
        "EBM (additive)": (lambda: ExplainableBoostingScratch(n_rounds=800, learning_rate=0.02, max_bins=16, max_leaves=2, min_samples_leaf=60,
                                                                    reg_lambda=20.0, n_outer_bags=8, n_interactions=0, linear_base=True, random_state=SEED), True),
        # ---- the better / extracurricular model: EBM + bagged MLP, fixed 50/50 blend (see BlendEnsemble.py) ----
        "Blend EBM+MLP (better model)": (lambda: BlendEnsembleScratch(n_mlps=5, ebm_weight=0.5, random_state=SEED), True),
    }


def fit_model(factory, use_val, X_tr, y_tr, X_va, y_va, names):
    m = factory()
    if use_val:
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], feature_names=names, early_stopping_rounds=30)
    else:
        m.fit(X_tr, y_tr, feature_names=names)
    return m


# -----------------------------------------------------------------------------
# 1. Data analysis
# -----------------------------------------------------------------------------
section("1. DATA ANALYSIS")
df = load_dataset()
X, y, feature_cols = build_classification_data(df)
print(f"Rows x columns : {df.shape}   |   missing values: {int(df.isnull().sum().sum())}")
print(f"Features used  : {len(feature_cols)}   |   positives (not.fully.paid=1): {y.mean():.2%}  <- imbalanced")
corr = pd.Series({c: np.corrcoef(X[:, i], y)[0, 1] for i, c in enumerate(feature_cols)})
corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
corr.rename("correlation_with_not.fully.paid").to_csv(out("data", "classification_correlation_with_target.csv"), index_label="feature")
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar(["paid (0)", "not fully paid (1)"], [(y == 0).sum(), (y == 1).sum()], color=[plots.BLUE, plots.RED])
ax.set_title("Class balance of not.fully.paid")
plots._save(fig, out("data", "classification_class_balance.png"))
top = corr.head(10)[::-1]
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(top.index, top.values, color=[plots.RED if v > 0 else plots.BLUE for v in top.values])
ax.set_title("Top correlations with not.fully.paid")
plots._save(fig, out("data", "classification_correlation_with_target.png"))
print(corr.head(6).to_string())

# -----------------------------------------------------------------------------
# 2. Split (train / validation / test)
# -----------------------------------------------------------------------------
section("2. TRAIN / VALIDATION / TEST SPLIT (60/20/20)")
X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test_split_scratch(X, y, seed=SEED)
print(f"Train {X_tr.shape}   Val {X_va.shape}   Test {X_te.shape}")

# -----------------------------------------------------------------------------
# 3. Train the 5 scratch models
# -----------------------------------------------------------------------------
section("3. TRAINING THE 11 SCRATCH MODELS")
models = make_models()
fitted, p_va, p_te, rows = {}, {}, {}, []
for name, (factory, use_val) in models.items():
    t0 = time.time()
    m = fit_model(factory, use_val, X_tr, y_tr, X_va, y_va, feature_cols)
    fitted[name] = m
    p_va[name] = m.predict_proba(X_va)[:, 1]
    p_te[name] = m.predict_proba(X_te)[:, 1]
    rep = evaluate_on_test(y_va, p_va[name], y_te, p_te[name])
    rows.append({"Model": name, **rep})
    print(f"  trained {name:<22} in {time.time() - t0:5.1f}s   test ROC-AUC {rep['test_roc_auc']:.3f}")
bench = pd.DataFrame(rows).set_index("Model")
# sensitivity (= recall) and specificity (= true-negative rate) from the confusion-matrix counts
bench["test_specificity"] = bench["test_tn"] / (bench["test_tn"] + bench["test_fp"])
bench["test_balanced_accuracy"] = (bench["test_recall"] + bench["test_specificity"]) / 2

# -----------------------------------------------------------------------------
# 4. Loss
# -----------------------------------------------------------------------------
section("4. LOSS (log loss)")
loss_table = bench[["test_logloss"]].rename(columns={"test_logloss": "Test log loss"})
loss_table.to_csv(out("loss", "classification_loss_table.csv"))
print(loss_table.to_string())
fig, ax = plt.subplots(figsize=(9, 4.5))
for name, m in fitted.items():
    if getattr(m, "train_loss", None):
        ax.plot(m.train_loss, label=name, color=COLORS[name])
ax.set_xlabel("iteration"); ax.set_ylabel("train log loss"); ax.set_title("Training loss curves"); ax.legend()
plots._save(fig, out("loss", "classification_loss_curves.png"))

# -----------------------------------------------------------------------------
# 5. Confusion matrix, accuracy, precision / recall / F1, ROC-AUC
# -----------------------------------------------------------------------------
section("5. CONFUSION MATRIX / ACCURACY / PRECISION-RECALL-F1 / ROC-AUC  (threshold chosen on validation)")
print(bench[["chosen_threshold", "test_accuracy", "test_precision", "test_recall", "test_specificity", "test_f1", "test_roc_auc", "test_pr_auc"]].to_string())
bench.to_csv(out("other", "classification_benchmark_all_metrics.csv"))
bench[["test_accuracy", "test_accuracy@0.5"]].to_csv(out("acc", "classification_accuracy.csv"))
bench[["test_precision", "test_recall", "test_f1"]].to_csv(out("prf", "classification_precision_recall_f1.csv"))
bench[["test_recall", "test_specificity", "test_balanced_accuracy"]].rename(columns={"test_recall": "sensitivity (recall)", "test_specificity": "specificity (TNR)", "test_balanced_accuracy": "balanced accuracy"}).to_csv(out("prf", "classification_sensitivity_specificity.csv"))
bench[["test_roc_auc", "test_pr_auc"]].to_csv(out("roc", "classification_roc_auc.csv"))

def bar(col, title, topic, fname, ylabel):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(bench.index, bench[col], color=[COLORS[n] for n in bench.index])
    ax.set_title(title); ax.set_ylabel(ylabel); plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plots._save(fig, out(topic, fname))

bar("test_accuracy", "Test accuracy (at the validation-chosen threshold)", "acc", "classification_accuracy.png", "accuracy")
bar("test_roc_auc", "Test ROC-AUC", "roc", "classification_roc_auc_bars.png", "ROC-AUC")

fig, ax = plt.subplots(figsize=(10, 4))
w = 0.26
for k, (col, lab) in enumerate([("test_precision", "precision"), ("test_recall", "recall"), ("test_f1", "F1")]):
    ax.bar(np.arange(len(bench)) + (k - 1) * w, bench[col], w, label=lab)
ax.set_xticks(range(len(bench))); ax.set_xticklabels(bench.index, rotation=30, ha="right"); ax.legend()
ax.set_title("Precision / recall / F1 on the test set")
plots._save(fig, out("prf", "classification_precision_recall_f1.png"))

fig, ax = plt.subplots(figsize=(9, 4))
for k, (col, lab) in enumerate([("test_recall", "sensitivity (defaulters caught)"), ("test_specificity", "specificity (good payers kept)")]):
    ax.bar(np.arange(len(bench)) + (k - 0.5) * 0.38, bench[col], 0.38, label=lab)
ax.set_xticks(range(len(bench))); ax.set_xticklabels(bench.index, rotation=30, ha="right"); ax.legend()
ax.set_title("Sensitivity vs specificity on the test set (validation-chosen threshold)")
plots._save(fig, out("prf", "classification_sensitivity_specificity.png"))

ncol = 5; nrow = (len(fitted) + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.4 * nrow)); axes = np.atleast_1d(axes).ravel()
for ax in axes[len(fitted):]:
    ax.axis("off")
for ax, name in zip(axes, fitted):
    cm = confusion_matrix(y_te, p_te[name], bench.loc[name, "chosen_threshold"])
    ax.imshow(cm, cmap="Blues"); ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"]); ax.set_yticklabels(["true 0", "true 1"])
    ax.set_title(name, fontsize=9)
plots._save(fig, out("cm", "classification_confusion_matrices.png"))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
for name in fitted:
    fpr, tpr, _ = roc_curve_scratch(y_te, p_te[name])
    a1.plot(fpr, tpr, color=COLORS[name], label=f"{name} ({bench.loc[name, 'test_roc_auc']:.3f})")
    pr, rc, _ = precision_recall_curve_scratch(y_te, p_te[name])
    a2.plot(rc, pr, color=COLORS[name], label=name)
a1.plot([0, 1], [0, 1], color=plots.NEUTRAL, linestyle="--"); a1.set_title("ROC curve"); a1.set_xlabel("false positive rate"); a1.set_ylabel("true positive rate"); a1.legend()
a2.axhline(y_te.mean(), color=plots.NEUTRAL, linestyle="--"); a2.set_title("Precision-recall curve"); a2.set_xlabel("recall"); a2.set_ylabel("precision")
plots._save(fig, out("roc", "classification_roc_and_pr_curves.png"))

# -----------------------------------------------------------------------------
# 6. Performance curve (validation log loss vs. boosting rounds)
# -----------------------------------------------------------------------------
section("6. PERFORMANCE CURVE")
fig, ax = plt.subplots(figsize=(7, 4))
for name, m in fitted.items():
    hist = getattr(m, "evals_result_", {}).get("validation", {}).get("logloss") if getattr(m, "evals_result_", None) else None
    if hist:
        ax.plot(hist, label=name, color=COLORS[name])
        print(f"  {name:<22} best validation log loss {min(hist):.4f} at iteration {int(np.argmin(hist)) + 1} of {len(hist)}")
ax.set_xlabel("iteration"); ax.set_ylabel("validation log loss"); ax.set_title("Validation loss while training"); ax.legend()
plots._save(fig, out("perf", "classification_validation_curves.png"))

# -----------------------------------------------------------------------------
# 7. 5-fold cross-validation (ROC-AUC)
# -----------------------------------------------------------------------------
section("7. 5-FOLD CROSS-VALIDATION  (test ROC-AUC per fold)")
cv_scores = {n: [] for n in models}
for k, (tr, te) in enumerate(kfold_indices(len(y), 5, SEED), 1):
    cut = int(len(tr) * 0.8)                      # inner validation slice for early stopping
    for name, (factory, use_val) in models.items():
        m = fit_model(factory, use_val, X[tr[:cut]], y[tr[:cut]], X[tr[cut:]], y[tr[cut:]], feature_cols)
        cv_scores[name].append(roc_auc(y[te], m.predict_proba(X[te])[:, 1]))
    print(f"  fold {k}/5 done")
cv = pd.DataFrame({"CV ROC-AUC mean": {n: np.mean(v) for n, v in cv_scores.items()},
                   "CV ROC-AUC std": {n: np.std(v) for n, v in cv_scores.items()}})
pd.DataFrame(cv_scores, index=[f"fold {k}" for k in range(1, 6)]).to_csv(out("other", "classification_cv_folds.csv"), index_label="fold")
print(cv.to_string())
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(cv.index, cv["CV ROC-AUC mean"], yerr=cv["CV ROC-AUC std"], color=[COLORS[n] for n in cv.index], capsize=4)
ax.set_ylim(0.4, 0.8); ax.set_title("5-fold CV ROC-AUC (mean +/- std)"); plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
plots._save(fig, out("other", "classification_cv_roc_auc.png"))

# -----------------------------------------------------------------------------
# 8. Feature importance (best tree model)
# -----------------------------------------------------------------------------
section("8. FEATURE IMPORTANCE")
imp_name = "XGBoost"
imp = pd.Series(dict(fitted[imp_name].feature_importance_table()))
imp.rename("importance").to_csv(out("other", "classification_feature_importance.csv"), index_label="feature")
print(imp.head(8).to_string())
top = imp.head(10)[::-1]
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(top.index, top.values, color=COLORS[imp_name]); ax.set_title(f"Feature importance ({imp_name}, gain share)")
plots._save(fig, out("other", "classification_feature_importance.png"))

# -----------------------------------------------------------------------------
# 9. Save + load the best model (highest validation PR-AUC)
# -----------------------------------------------------------------------------
section("9. SAVE / LOAD THE BEST MODEL")
best_name = max(fitted, key=lambda n: average_precision(y_va, p_va[n]))
os.makedirs(os.path.join(ROOT, "TrainedModels"), exist_ok=True)
path = os.path.join(ROOT, "TrainedModels", "classification_model.joblib")
thr = float(bench.loc[best_name, "chosen_threshold"])
joblib.dump({"model": fitted[best_name], "name": best_name, "feature_cols": feature_cols, "target": "not.fully.paid",
             "threshold": thr, "test_roc_auc": float(bench.loc[best_name, "test_roc_auc"])}, path)
bundle = joblib.load(path)
assert np.allclose(bundle["model"].predict_proba(X_te)[:, 1], p_te[best_name]), "reloaded model differs!"
print(f"Best model by validation PR-AUC: {best_name}")
print(f"Saved + reloaded: {os.path.relpath(path, ROOT)}  (threshold {thr:.3f}, test ROC-AUC {bundle['test_roc_auc']:.4f})")
risk = float(bundle["model"].predict_proba(X_te[:1])[0, 1])
print(f"Example borrower #1 of the test set: P(not fully paid) = {risk:.3f} -> "
      f"{'HIGH RISK' if risk >= thr else 'low risk'} (threshold {thr:.3f}); true label = {int(y_te[0])}")
# -----------------------------------------------------------------------------
# 10. Part 2 extras: scratch vs scikit-learn, perceptron variants, EBM curves, Naive Bayes, explain one borrower
# -----------------------------------------------------------------------------
section("10. PART 2 EXTRAS (verification + interpretability)")
w_tr = np.where(y_tr == 1, np.sqrt((len(y_tr) - y_tr.sum()) / y_tr.sum()), 1.0)      # project imbalance weight
ver = []
try:
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import CategoricalNB, GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler as _SK

    g_s = NaiveBayesScratch(kind="gaussian").fit(X_tr, y_tr, feature_names=feature_cols)
    g_k = GaussianNB().fit(X_tr, y_tr)
    ver.append(("Naive Bayes (Gaussian) vs GaussianNB", float(np.abs(g_s.predict_proba(X_te)[:, 1] - g_k.predict_proba(X_te)[:, 1]).max()),
                roc_auc(y_te, g_s.predict_proba(X_te)[:, 1]), roc_auc(y_te, g_k.predict_proba(X_te)[:, 1]), "max |dP|"))
    nbc = fitted["Naive Bayes"]
    binned = lambda A: np.column_stack([np.searchsorted(nbc.edges_[j], A[:, j], side="right") for j in range(A.shape[1])])
    c_k = CategoricalNB(alpha=1.0, min_categories=[len(e) + 1 for e in nbc.edges_]).fit(binned(X_tr), y_tr)
    ver.append(("Naive Bayes (categorical) vs CategoricalNB", float(np.abs(p_te["Naive Bayes"] - c_k.predict_proba(binned(X_te))[:, 1]).max()),
                roc_auc(y_te, p_te["Naive Bayes"]), roc_auc(y_te, c_k.predict_proba(binned(X_te))[:, 1]), "max |dP|"))
    sc = _SK().fit(X_tr)
    p_sk = SGDClassifier(loss="perceptron", penalty=None, learning_rate="constant", eta0=0.1, average=True, max_iter=40, tol=None,
                         class_weight={0: 1, 1: float(w_tr.max())}, random_state=SEED).fit(sc.transform(X_tr), y_tr)
    ver.append(("Averaged perceptron vs SGDClassifier(perceptron, average)", float("nan"),
                roc_auc(y_te, fitted["Perceptron (averaged)"].decision_function(X_te)), roc_auc(y_te, p_sk.decision_function(sc.transform(X_te))), "AUC only"))
    l2 = 1e-3
    slp_full = SingleLayerPerceptronScratch(learning_rate=0.05, n_epochs=400, l2=l2, lr_decay=0.02, random_state=SEED).fit(X_tr, y_tr, early_stopping_rounds=0)
    lr_same = LogisticRegression(C=1 / (l2 * w_tr.sum()), max_iter=5000, tol=1e-10).fit(sc.transform(X_tr), y_tr, sample_weight=w_tr)
    ver.append(("Single-layer perceptron vs LogisticRegression (same loss)", float(np.abs(slp_full.coef_ - lr_same.coef_[0]).max()),
                roc_auc(y_te, slp_full.predict_proba(X_te)[:, 1]), roc_auc(y_te, lr_same.predict_proba(sc.transform(X_te))[:, 1]), "max |dcoef|"))
    mlp_sk = MLPClassifier((64, 32), alpha=1e-2, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, max_iter=300, random_state=SEED)
    try:
        mlp_sk.fit(sc.transform(X_tr), y_tr, sample_weight=w_tr)
    except TypeError:
        mlp_sk.fit(sc.transform(X_tr), y_tr)
    ver.append(("MLP vs sklearn MLPClassifier (same shape)", float("nan"), roc_auc(y_te, p_te["MLP"]), roc_auc(y_te, mlp_sk.predict_proba(sc.transform(X_te))[:, 1]), "AUC only"))
    ver.append(("MLP back-propagation vs numerical gradient", MLPClassifierScratch(hidden_layers=(16, 8)).gradient_check(X_tr, y_tr, n_checks=30), float("nan"), float("nan"), "max relative error"))
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
        ref = ExplainableBoostingClassifier(interactions=0, random_state=SEED, outer_bags=4, learning_rate=0.02, max_bins=16).fit(X_tr, y_tr, sample_weight=w_tr)
        pr = ref.predict_proba(X_te)[:, 1]
        ver.append(("EBM vs interpret.ExplainableBoostingClassifier", float(np.corrcoef(p_te["EBM (additive)"], pr)[0, 1]), roc_auc(y_te, p_te["EBM (additive)"]), roc_auc(y_te, pr), "probability correlation"))
    except Exception:
        pass
except ImportError:
    print("  scikit-learn is not installed: skipping the scratch-vs-sklearn checks (pip install scikit-learn)")
if ver:
    vt = pd.DataFrame(ver, columns=["check", "difference", "AUC scratch", "AUC sklearn", "difference means"]).set_index("check")
    vt.to_csv(out("other", "classification_scratch_vs_sklearn.csv"))
    print(vt.to_string())
    pairs = vt.dropna(subset=["AUC scratch", "AUC sklearn"])
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = np.arange(len(pairs))
    ax.bar(xs - 0.2, pairs["AUC scratch"], 0.4, label="scratch (ours)", color=plots.BLUE); ax.bar(xs + 0.2, pairs["AUC sklearn"], 0.4, label="scikit-learn", color=plots.NEUTRAL)
    ax.set_xticks(xs); ax.set_xticklabels([c.split(" vs ")[0] for c in pairs.index], rotation=20, ha="right"); ax.set_ylim(0.5, 0.75); ax.legend()
    ax.set_title("Scratch models vs scikit-learn counterparts (test ROC-AUC)")
    plots._save(fig, out("other", "classification_scratch_vs_sklearn.png"))

# perceptron variants: the classic rule never converges on non-separable data, averaging fixes it
pv_rows = []
for v in ("classic", "pocket", "averaged"):
    pm = PerceptronScratch(variant=v, learning_rate=0.1, n_epochs=40, random_state=SEED).fit(X_tr, y_tr, eval_set=[(X_va, y_va)], feature_names=feature_cols)
    pv_rows.append({"variant": v, "val ROC-AUC": roc_auc(y_va, pm.decision_function(X_va)), "test ROC-AUC": roc_auc(y_te, pm.decision_function(X_te)),
                    "mistakes in last epoch": pm.history_["mistakes"][-1], "converged": pm.converged_})
pv = pd.DataFrame(pv_rows).set_index("variant"); pv.to_csv(out("other", "classification_perceptron_variants.csv")); print("\n" + pv.to_string())

# EBM: one readable curve per feature + importance
ebm = fitted["EBM (additive)"]
ebm_imp = pd.Series(dict(ebm.feature_importance_table()))
ebm_imp.rename("importance").to_csv(out("other", "classification_ebm_feature_importance.csv"), index_label="feature")
top_f = [feature_cols.index(f) for f in ebm_imp.index[:8]]
fig, axes = plt.subplots(2, 4, figsize=(16, 6.5)); axes = axes.ravel()
for ax, j in zip(axes, top_f):
    c, t, sd = ebm.shape_function(j, with_std=True)
    ax.step(c, t, where="mid", color=plots.RED); ax.fill_between(c, t - 2 * sd, t + 2 * sd, step="mid", color=plots.RED, alpha=0.15)
    ax.axhline(0, color=plots.NEUTRAL, linewidth=0.8); ax.set_title(f"{feature_cols[j]} ({ebm.feature_importances_[j]:.0%})", fontsize=9); ax.set_ylabel("log-odds of default")
fig.suptitle("EBM shape functions: effect of each feature on the log-odds of default (band = +-2 SD across bags)")
plots._save(fig, out("other", "classification_ebm_shape_functions.png"))

# Naive Bayes: the per-class Gaussians that get multiplied together
nbg = NaiveBayesScratch(kind="gaussian").fit(X_tr, y_tr, feature_names=feature_cols)
topn = np.argsort(-nbg.feature_importances_)[:4]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
for ax, j in zip(axes, topn):
    xs_ = np.linspace(np.percentile(X_tr[:, j], 0.5), np.percentile(X_tr[:, j], 99.5), 300)
    for c_, col, lab in ((0, plots.BLUE, "paid"), (1, plots.RED, "defaulted")):
        ax.hist(X_tr[y_tr == c_, j], bins=30, range=(xs_[0], xs_[-1]), density=True, alpha=0.25, color=col)
        ax.plot(xs_, np.exp(-0.5 * (xs_ - nbg.theta_[c_, j]) ** 2 / nbg.var_[c_, j]) / np.sqrt(2 * np.pi * nbg.var_[c_, j]), color=col, label=lab)
    ax.set_title(feature_cols[j], fontsize=9); ax.legend(fontsize=7)
fig.suptitle("Naive Bayes: fitted Gaussian per class (bars = real data)")
plots._save(fig, out("other", "classification_naive_bayes_densities.png"))

# save the better model in its own file, reload it, explain one borrower
os.makedirs(os.path.join(ROOT, "TrainedModels"), exist_ok=True)
ebm_path = os.path.join(ROOT, "TrainedModels", "classification_ebm_model.joblib")
ebm.best_threshold_ = float(bench.loc["EBM (additive)", "chosen_threshold"]); ebm.save(ebm_path)
ebm2 = ExplainableBoostingScratch.load(ebm_path)
assert np.allclose(ebm2.predict_proba(X_te)[:, 1], p_te["EBM (additive)"]), "reloaded EBM differs!"
flagged = np.where((y_te == 1) & (ebm2.predict(X_te) == 1))[0]
i = int(flagged[0]) if len(flagged) else 0
exp = ebm2.explain(X_te[i:i + 1], top=6)
bl = fitted["Blend EBM+MLP (better model)"]
bl_path = os.path.join(ROOT, "TrainedModels", "classification_blend_model.joblib")
bl.best_threshold_ = float(bench.loc["Blend EBM+MLP (better model)", "chosen_threshold"]); bl.save(bl_path)
bl2 = BlendEnsembleScratch.load(bl_path)
assert np.allclose(bl2.predict_proba(X_te)[:, 1], p_te["Blend EBM+MLP (better model)"]), "reloaded blend differs!"
ex = bl2.explain(X_te[i:i + 1])
print(f"\nSaved + reloaded: {os.path.relpath(bl_path, ROOT)}  (reload gives identical predictions)")
print(f"Same borrower with the blend: P(EBM) = {ex['p_ebm']:.3f}, P(bagged MLP) = {ex['p_mlp_bag']:.3f} -> P(blend) = {ex['p_blend']:.3f}")
print(f"\nSaved + reloaded: {os.path.relpath(ebm_path, ROOT)}")
print(f"Borrower #{i} of the test set (true label {int(y_te[i])}): P(default) = {ebm2.predict_proba(X_te[i:i + 1])[0, 1]:.3f}  (threshold {ebm2.best_threshold_:.3f})")
print(f"  log-odds = intercept {exp['intercept']:+.3f} + contributions (sum = {exp['log_odds']:+.3f}):")
for f_, v_ in exp["contributions"]:
    print(f"     {f_:<32} {v_:+.3f}")

# -----------------------------------------------------------------------------
# 11. Batch runs of the better model: 10 fresh random 60/20/20 splits (seeds never used to design the blend)
# -----------------------------------------------------------------------------
section("11. BATCH RUNS: 10 RANDOM SPLITS, BETTER MODEL vs TAUGHT MODELS  (test ROC-AUC / PR-AUC, mean +/- SD)")
BATCH_MODELS = {"Logistic Regression": models["Logistic Regression"], "Random Forest": models["Random Forest"],
                "Gradient Boosting": models["Gradient Boosting"], "MLP": models["MLP"], "EBM (additive)": models["EBM (additive)"]}
bres = {n: {"auc": [], "pr": []} for n in list(BATCH_MODELS) + ["Blend EBM+MLP (better model)"]}
for b in range(10):
    s_tr, s_va, s_te, t_tr, t_va, t_te = train_val_test_split_scratch(X, y, seed=500 + b)
    ps = {}
    for n, (factory, use_val) in BATCH_MODELS.items():
        ps[n] = fit_model(factory, use_val, s_tr, t_tr, s_va, t_va, feature_cols).predict_proba(s_te)[:, 1]
    bm = BlendEnsembleScratch(n_mlps=5, ebm_weight=0.5, random_state=SEED).fit(s_tr, t_tr, eval_set=[(s_va, t_va)], feature_names=feature_cols)
    ps["Blend EBM+MLP (better model)"] = bm.predict_proba(s_te)[:, 1]
    for n, p_ in ps.items():
        bres[n]["auc"].append(roc_auc(t_te, p_)); bres[n]["pr"].append(average_precision(t_te, p_))
    print(f"  split {b + 1}/10 done")
bt = pd.DataFrame({n: {"ROC-AUC mean": np.mean(v["auc"]), "ROC-AUC SD": np.std(v["auc"]), "PR-AUC mean": np.mean(v["pr"]), "PR-AUC SD": np.std(v["pr"])} for n, v in bres.items()}).T
bt.to_csv(out("other", "classification_batch_runs.csv"), index_label="Model"); print(bt.to_string())
bl_auc = np.array(bres["Blend EBM+MLP (better model)"]["auc"])
for n in BATCH_MODELS:
    dlt = bl_auc - np.array(bres[n]["auc"])
    print(f"  Blend minus {n:<20}: {dlt.mean():+.4f} AUC (SD {dlt.std():.4f}), better in {(dlt > 0).sum()}/10 splits")
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
for ax, k, t in ((axes[0], "auc", "ROC-AUC"), (axes[1], "pr", "PR-AUC")):
    bp = ax.boxplot([bres[n][k] for n in bres], tick_labels=list(bres), patch_artist=True, showmeans=True)
    for patch_, n in zip(bp["boxes"], bres):
        patch_.set_facecolor(COLORS[n]); patch_.set_alpha(0.7)
    ax.set_title(f"Test {t} over 10 random splits"); plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
plots._save(fig, out("other", "classification_batch_runs.png"))

print("\nDONE.")
