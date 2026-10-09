"""
Run the whole CLASSIFICATION task (target: not.fully.paid) -- the twin of run_regression.py.

  load data -> data analysis -> split train/validation/test -> pick the better model's blend weight by CV on TRAIN
  -> train the 15 scratch models -> loss -> thresholds aimed at CATCHING DEFAULTERS
  -> test metrics (accuracy, precision/recall/F1, ROC-AUC, PR-AUC, confusion matrix) -> curves for every model
  -> scratch-vs-scikit-learn verification for every model -> 5-fold cross-validation -> benchmark table
  -> feature importance for every model -> save + load the best model and score example borrowers
  -> model-specific interpretability -> 10 batch runs over every model -> list every result file.

Goal of this half of the project: catch the defaulters (the 16% positive class) without flagging everybody.
Ranking quality is judged threshold-free by ROC-AUC / PR-AUC / recall@top-k; the flagging decision uses a
threshold tuned on the VALIDATION set only, and section 6 prints the full rule ladder so the trade-off is
visible on the data: a plain 0.5 catches ~14% of the defaulters, f1 ~67% while flagging ~44% of applicants,
f2 ~91% but flagging ~76% of ALL applicants (enrichment barely 1.19x). The rule in use is the one-word
constant FLAG_RULE at the top of this file.

Run:  python run_classification.py        (from the project root)
Outputs: printed tables; Results/<topic folders>/ (CSV tables + PNG charts); TrainedModels/*.joblib.
Charts that matter here: classification_loss_curves_all_models.png (every model on one axis),
classification_threshold_rules.png + _ladder.csv (what each flagging rule costs), classification_recall_at_topk.png
(defaulters caught when the riskiest x% is reviewed). Set CLF_QUICK=1 for a ~8-minute smoke run while editing.

Section map (the topics of "Quantitative results, benchmarking, analysis & discussion"):
   1  Data analysis                                            01_Data_Analysis
   2  Train / validation / test split
   3  Hyper-parameter search of the better model (3-fold CV on the train set)
   4  Train the 15 scratch models
   5  Loss (train / validation / test log loss, curves)        02_Loss
   6  Catching defaulters: rule ladder + recall@top-k           09_Others
   7  Confusion matrix, accuracy, precision / recall / F1      03_Confusion_Matrix, 04_Accuracy, 05_Precision_Recall_F1
   8  ROC-AUC, PR-AUC and the ROC / PR curves                  06_ROC_AUC
   9  Performance curve of every staged model                  07_Performance_Curve
  10  Extra check: scratch vs scikit-learn, every model        09_Others
  11  5-fold cross-validation (stratified)                     09_Others
  12  Benchmark table (every model, every metric)              09_Others
  13  Feature importance for every model                       09_Others
  14  Save + load the best model, score example borrowers       TrainedModels/
  15  Model-specific interpretability (perceptron, EBM, NB)     09_Others
  16  Batch runs: 10 fresh random splits, every model           09_Others
  17  Checklist of every file written
"""
import os
import re
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
from data_classification import build_classification_data, load_dataset, train_val_test_split_scratch
from DecisionTree import DecisionTreeClassifierScratch
from BlendEnsemble import BlendEnsembleScratch
from ExplainableBoosting import ExplainableBoostingScratch
from GradientBoosting import GradientBoostingClassifierScratch
from KNN import KNNClassifierScratch
from Logisticregression import LogisticRegressionScratch
from MLP import MLPClassifierScratch
from NaiveBayes import NaiveBayesScratch
from Perceptron import PerceptronScratch, SingleLayerPerceptronScratch
from metrics_classification import (accuracy, average_precision, classification_metrics, f1, logloss,
                                    precision, precision_recall_curve_scratch, recall, roc_auc,
                                    roc_curve_scratch, confusion_matrix)
from RandomForest import RandomForestClassifierScratch
from SVC import SVCScratch
from Stackingclassifier import StackingClassifierScratch, default_base_models
from XGB import XGBClassifierScratch

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")
pd.set_option("display.show_dimensions", False)
SEED = 42
RESULTS = os.path.join(ROOT, "Results")
FOLDERS = {"data": "01_Data_Analysis", "loss": "02_Loss", "cm": "03_Confusion_Matrix", "acc": "04_Accuracy",
           "prf": "05_Precision_Recall_F1", "roc": "06_ROC_AUC", "perf": "07_Performance_Curve", "other": "09_Others"}
BETTER = "Blend EBM+MLP (better model)"
# One-word switch for the flagging rule (section 6 compares all of them on the data):
#   "f1"   balanced   - catches about 2/3 of the defaulters while flagging under half of the applicants
#   "f1.5" middle     - one notch more recall than f1, still nowhere near "flag everybody"      <- default
#   "f2"   catch-max  - catches ~90% of defaulters but flags ~76% of ALL applicants (enrichment only 1.19x)
#   "precision>=1.5x base" - conservative: flag only while the flagged group stays 1.5x enriched
FLAG_RULE = "f1.5"
COLORS = {"Logistic Regression": "#2a78d6", "Decision Tree": "#eb6834", "Random Forest": "#1baf7a",
          "Gradient Boosting": "#eda100", "XGBoost": "#8b5cf6",
          "SVC (linear)": "#00a3a3", "SVC (rbf)": "#00838f", "k-NN": "#c2185b",
          "Naive Bayes": "#d4549a", "Perceptron (averaged)": "#0f9bb3", "Single-Layer Perceptron": "#9aa31a",
          "MLP": "#8d6e63", "EBM (additive)": "#111111", "Stacking (5 bases)": "#7b1fa2", BETTER: "#d62728"}
EBM_WEIGHT, N_MLPS = 0.5, 5          # overwritten by the validation search in section 3
TOPK = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]     # flag the riskiest 5% ... 50% of the applicants
# Quick check while editing (never for the report): CLF_QUICK=1 run_classification.py
QUICK = os.environ.get("CLF_QUICK") == "1"
N_BATCH = 1 if QUICK else 10                     # 10 fresh random 60/20/20 splits in section 16
CV_FOLDS = 2 if QUICK else 5                     # stratified folds in section 11
SEARCH_FOLDS = 2 if QUICK else 3                       # CV folds used to choose the blend weight
SEARCH_WEIGHTS = (0.2, 0.7) if QUICK else (0.0, 0.2, 0.3, 0.5, 0.7)


def out(topic, filename):
    folder = os.path.join(RESULTS, FOLDERS[topic])
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)


def section(title):
    print("\n" + "=" * 78 + f"\n  {title}\n" + "=" * 78)


# -----------------------------------------------------------------------------
# threshold helpers: the flagging rule is tuned to CATCH DEFAULTERS, on validation only
# -----------------------------------------------------------------------------
def _counts(y_true, proba, thr):
    """tp, fp, fn, tn at one threshold."""
    y = np.asarray(y_true, dtype=int).ravel()
    pred = (np.asarray(proba, dtype=float).ravel() >= thr).astype(int)
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    tn = float(np.sum((pred == 0) & (y == 0)))
    return tp, fp, fn, tn


def f_beta(y_true, proba, thr, beta=2.0):
    """F-beta score at one threshold. beta=1 -> F1, beta=2 -> recall counts twice as much as precision."""
    tp, fp, fn, _ = _counts(y_true, proba, thr)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec) if prec + rec else 0.0


def youden_j(y_true, proba, thr):
    """sensitivity + specificity - 1 (the ROC point furthest above the diagonal)."""
    tp, fp, fn, tn = _counts(y_true, proba, thr)
    return (tp / (tp + fn) if tp + fn else 0.0) + (tn / (tn + fp) if tn + fp else 0.0) - 1.0


def choose_threshold(y_true, proba, rule=FLAG_RULE, base_rate=None):
    """Scan the thresholds and return the one that maximises `rule`, using the VALIDATION scores only.

    "f<beta>"   - maximum F-beta: "f1" (balanced), "f1.5", "f2" (recall counts twice)     <- the headline rule
    "precision>=<k>x base" - highest recall whose precision stays at least k x the base rate, i.e. the
                             flagged group must stay k times more likely to default than a random applicant
    "youden"    - sensitivity + specificity - 1, the classic ROC-based rule
    "accuracy"  - the naive reference (predicts almost everybody as a good payer on 16% positives)
    """
    p = np.asarray(proba, dtype=float).ravel()
    grid = np.unique(np.concatenate([np.linspace(0.001, 0.999, 499), p]))
    base = float(np.mean(y_true)) if base_rate is None else float(base_rate)
    m_fl = re.match(r"precision>=\s*([0-9.]+)\s*x", rule)
    floor = float(m_fl.group(1)) * base if m_fl else None
    m_fb = re.match(r"f([0-9.]+)$", rule)
    beta = float(m_fb.group(1)) if m_fb else None
    best_thr, best_score = 0.5, -np.inf
    for thr in grid:
        if floor is not None:
            score = recall(y_true, p, thr) if precision(y_true, p, thr) >= floor else -1.0
        elif beta is not None:
            score = f_beta(y_true, p, thr, beta)
        elif rule == "youden":
            score = youden_j(y_true, p, thr)
        else:
            score = accuracy(y_true, p, thr)
        if score > best_score:
            best_score, best_thr = float(score), float(thr)
    return best_thr, best_score


def evaluate_model(y_val, p_val, y_test, p_test, rule=FLAG_RULE, base_rate=None):
    """The honest one-call report, with the defaulter-catching rule as the headline:

      1. choose the flagging threshold on the VALIDATION set (rule = FLAG_RULE, f1.5 by default)
      2. report every test metric at that threshold
      3. also report the 0.5 and the F1-rule references, so the trade-off is visible and never hidden
    """
    thr, val_score = choose_threshold(y_val, p_val, rule=rule, base_rate=base_rate)
    row = {"chosen_threshold": float(thr), "val_" + rule: float(val_score), "threshold_rule": rule}
    cm = classification_metrics(y_test, p_test, threshold=thr)
    for k in ("logloss", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"):
        row["test_" + k] = cm[k]
    row["test_threshold"] = float(thr)
    row["test_threshold_auto_selected"] = False
    for k, fn_ in (("accuracy", accuracy), ("precision", precision), ("recall", recall), ("f1", f1)):
        row[f"test_{k}@0.5"] = float(fn_(y_test, p_test, 0.5))
    thr_f1, _ = choose_threshold(y_val, p_val, rule="f1")
    row["thr_f1_rule"] = float(thr_f1)
    row["test_recall@f1_rule"] = float(recall(y_test, p_test, thr_f1))
    row["test_precision@f1_rule"] = float(precision(y_test, p_test, thr_f1))
    thr_pr, _ = choose_threshold(y_val, p_val, rule="youden")
    row["thr_youden_rule"] = float(thr_pr)
    row["test_recall@youden_rule"] = float(recall(y_test, p_test, thr_pr))
    row["test_precision@youden_rule"] = float(precision(y_test, p_test, thr_pr))
    return row


def topk_table(y_test, p_test, fracs=TOPK):
    """Rank the applicants by predicted risk and flag the riskiest x%: how many defaulters does that catch?"""
    y = np.asarray(y_test, dtype=int).ravel()
    p = np.asarray(p_test, dtype=float).ravel()
    order = np.argsort(-p)
    n, n_pos = len(y), int(y.sum())
    rows = []
    for f in fracs:
        k = max(1, int(round(f * n)))
        caught = int(y[order[:k]].sum())
        prec = caught / k
        rows.append({"flagged share": f, "applicants flagged": k, "defaulters caught": caught,
                     "recall (defaulters caught / all)": caught / n_pos,
                     "precision inside the flagged group": prec, "lift vs flagging at random": prec / (n_pos / n)})
    return rows


def make_models(ebm_weight=EBM_WEIGHT):
    """name -> (factory, uses_validation_set). Fresh model every call so CV folds never share state."""
    return {
        "Logistic Regression": (lambda: LogisticRegressionScratch(learning_rate=0.5, n_iterations=500, reg_lambda=1.0), True),
        "Decision Tree": (lambda: DecisionTreeClassifierScratch(max_depth=6, min_samples_leaf=20, random_state=SEED), False),
        "Random Forest": (lambda: RandomForestClassifierScratch(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=SEED), False),
        "Gradient Boosting": (lambda: GradientBoostingClassifierScratch(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED), True),
        "XGBoost": (lambda: XGBClassifierScratch(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=SEED), True),
        # ---- margin-based models (settings chosen on the VALIDATION set only) ----
        # linear hinge with the project imbalance weight; measured: without "balanced" the hinge gives up and
        # answers "everybody pays" (the trivial solution has a lower objective on this data)
        "SVC (linear)": (lambda: SVCScratch(C=1.0, kernel="linear", solver="primal", n_iterations=1500, scale_pos_weight="balanced", random_state=SEED), True),
        "SVC (rbf)": (lambda: SVCScratch(C=1.0, kernel="rbf", n_iterations=800, scale_pos_weight="balanced", random_state=SEED), True),
        "k-NN": (lambda: KNNClassifierScratch(n_neighbors=25, weights="distance", tune_k="auto", scale_pos_weight="balanced", random_state=SEED), True),
        # ---- part 2 (settings chosen on the VALIDATION set only; see docs/PART2_GUIDE.md) ----
        "Naive Bayes": (lambda: NaiveBayesScratch(kind="categorical", n_bins=10), False),
        "Perceptron (averaged)": (lambda: PerceptronScratch(variant="averaged", learning_rate=0.1, n_epochs=40, random_state=SEED), True),
        "Single-Layer Perceptron": (lambda: SingleLayerPerceptronScratch(learning_rate=0.05, n_epochs=300, l2=1e-3, lr_decay=0.3, random_state=SEED), True),
        "MLP": (lambda: MLPClassifierScratch(hidden_layers=(64, 32), l2=1e-2, dropout=0.2, learning_rate=1e-3, n_epochs=200, random_state=SEED), True),
        "EBM (additive)": (lambda: ExplainableBoostingScratch(n_rounds=800, learning_rate=0.02, max_bins=16, max_leaves=2, min_samples_leaf=60,
                                                              reg_lambda=20.0, n_outer_bags=8, n_interactions=0, linear_base=True, random_state=SEED), True),
        # ---- ensembles ----
        "Stacking (5 bases)": (lambda: StackingClassifierScratch(default_base_models(SEED), n_folds=5, random_state=SEED), True),
        # ---- the better / extracurricular model: EBM + bagged MLP, blend weight fixed by the validation search ----
        BETTER: (lambda: BlendEnsembleScratch(n_mlps=N_MLPS, ebm_weight=ebm_weight, random_state=SEED), True),
    }


def fit_model(factory, use_val, X_tr, y_tr, X_va, y_va, names):
    m = factory()
    if use_val:
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], feature_names=names, early_stopping_rounds=30)
    else:
        m.fit(X_tr, y_tr, feature_names=names)
    return m


def _staged_pos(model, X):
    """P(class 1) at every training stage. Some models yield a 1-D vector (Gradient Boosting, XGBoost),
    others yield the full (n, 2) probability matrix (Decision Tree, Random Forest, Logistic Regression)."""
    out = []
    for p in model.staged_predict_proba(X):
        a = np.asarray(p, dtype=float)
        out.append(a[:, 1] if a.ndim == 2 else a.ravel())
    return out


def stratified_folds(y, n_splits=5, seed=SEED):
    """k folds with (almost) the same positive rate in each -- kfold_indices() in Common is not stratified,
    and with a 16% positive class that matters. Returns [(train_idx, test_idx), ...]."""
    rng = np.random.RandomState(seed)
    buckets = [[] for _ in range(n_splits)]
    for cls in (0, 1):
        idx = np.where(np.asarray(y) == cls)[0]
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            buckets[i % n_splits].append(j)          # deal the shuffled class members round-robin
    folds = []
    for f in buckets:
        te = np.sort(np.asarray(f, dtype=int))
        folds.append((np.setdiff1d(np.arange(len(y)), te), te))
    return folds


# -----------------------------------------------------------------------------
# 1. Data analysis
# -----------------------------------------------------------------------------
section("1. DATA ANALYSIS  [artifact: Task / data analysis, feature insight]")
df = load_dataset()
X, y, feature_cols = build_classification_data(df)
print(f"Rows x columns : {df.shape}   |   missing values: {int(df.isnull().sum().sum())}")
print(f"Features used  : {len(feature_cols)}   |   positives (not.fully.paid=1): {y.mean():.2%}  <- imbalanced")
corr = pd.Series({c: np.corrcoef(X[:, i], y)[0, 1] for i, c in enumerate(feature_cols)})
corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
corr.rename("correlation_with_not.fully.paid").to_csv(out("data", "classification_correlation_with_target.csv"), index_label="feature")
pd.DataFrame({"statistic": ["rows", "columns", "missing_values", "features_used", "positive_class_rate",
                            "negatives", "positives", "int_rate_mean", "fico_mean", "dti_mean"],
              "value": [len(df), df.shape[1], int(df.isnull().sum().sum()), len(feature_cols), float(y.mean()),
                        int((y == 0).sum()), int((y == 1).sum()), float(df["int.rate"].mean()),
                        float(df["fico"].mean()), float(df["dti"].mean())]}).to_csv(
    out("data", "classification_summary_stats.csv"), index=False)

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar([f"paid (0)\n{(y == 0).sum()} rows", f"not fully paid (1)\n{(y == 1).sum()} rows"],
       [(y == 0).sum(), (y == 1).sum()], color=[plots.BLUE, plots.RED])
ax.set_ylabel("borrowers"); ax.set_title(f"Class balance of not.fully.paid ({y.mean():.1%} positives)")
plots._save(fig, out("data", "classification_class_balance.png"))

top = corr.head(10)[::-1]
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(top.index, top.values, color=[plots.RED if v > 0 else plots.BLUE for v in top.values])
ax.axvline(0, color=plots.NEUTRAL, linewidth=0.8)
ax.set_title("Top correlations with not.fully.paid"); ax.set_xlabel("Pearson correlation")
plots._save(fig, out("data", "classification_correlation_with_target.png"))

# correlation heatmap between the features themselves (which columns carry the same information)
vc = corr.head(12)
sub = [feature_cols.index(f) for f in vc.index]
M = np.corrcoef(X[:, sub].T)
fig, ax = plt.subplots(figsize=(7.5, 6.4))
im = ax.imshow(M, cmap=plots.DIVERGING, vmin=-1, vmax=1)
ax.set_xticks(range(len(sub)), [c[:18] for c in vc.index], rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(sub)), [c[:18] for c in vc.index], fontsize=8)
for (i, j), v in np.ndenumerate(M):
    if i != j:
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                color="white" if abs(v) > 0.55 else "black")
fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title("Correlation between the 12 features most correlated with the target")
plots._save(fig, out("data", "classification_correlation_heatmap.png"))

# how the strongest features are distributed for the two classes
numeric_top = [c for c in corr.index if c in feature_cols and len(np.unique(X[:, feature_cols.index(c)])) > 2][:4]
fig, axes = plt.subplots(1, len(numeric_top), figsize=(3.6 * len(numeric_top), 3.4))
for ax, c in zip(np.atleast_1d(axes), numeric_top):
    j = feature_cols.index(c)
    bins = np.linspace(np.percentile(X[:, j], 0.5), np.percentile(X[:, j], 99.5), 30)
    ax.hist(X[y == 0, j], bins=bins, density=True, alpha=0.55, color=plots.BLUE, label="paid")
    ax.hist(X[y == 1, j], bins=bins, density=True, alpha=0.55, color=plots.RED, label="defaulted")
    ax.set_title(c, fontsize=9); ax.set_yticks([]); ax.legend(fontsize=7)
fig.suptitle("Distributions of the strongest features by class")
plots._save(fig, out("data", "classification_feature_distributions_by_class.png"))

# default rate per loan purpose: the business view (purpose is one-hot encoded in the model)
pur = df.groupby("purpose")["not.fully.paid"].agg(["mean", "size"]).sort_values("mean")
pur.columns = ["default_rate", "loans"]
pur["default_rate"] = pur["default_rate"].astype(float)
pur.to_csv(out("data", "classification_default_rate_by_purpose.csv"), index_label="purpose")
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.barh(pur.index, pur["default_rate"] * 100, color=plots.RED, height=0.6)
ax.axvline(float(y.mean()) * 100, color=plots.NEUTRAL, linestyle="--", label=f"overall {y.mean():.1%}")
for name, v, n in zip(pur.index, pur["default_rate"] * 100, pur["loans"]):
    ax.text(v + 0.2, name, f"{v:.2f}%  (n={n})", va="center", fontsize=8)
ax.set_xlim(0, pur["default_rate"].max() * 130); ax.grid(axis="y", visible=False)
ax.set_xlabel("default rate (%)"); ax.set_title("Default rate by loan purpose"); ax.legend()
plots._save(fig, out("data", "classification_default_rate_by_purpose.png"))

print("\nTop correlations with not.fully.paid:")
print(corr.head(6).to_string())
print("\nDefault rate by purpose (highest first):")
print(pur.sort_values("default_rate", ascending=False).to_string())

# -----------------------------------------------------------------------------
# 2. Split (train / validation / test)
# -----------------------------------------------------------------------------
section("2. TRAIN / VALIDATION / TEST SPLIT (60/20/20)")
X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test_split_scratch(X, y, seed=SEED)
print(f"Train {X_tr.shape}   Val {X_va.shape}   Test {X_te.shape}")
print(f"positives: train {y_tr.mean():.2%} ({int(y_tr.sum())}), val {y_va.mean():.2%} ({int(y_va.sum())}), "
      f"test {y_te.mean():.2%} ({int(y_te.sum())})")

# -----------------------------------------------------------------------------
# 3. Hyper-parameter search of the better model: the blend weight, chosen on VALIDATION only
#    (the same role as section 3 of run_regression.py; the search uses a 3-MLP bag to stay quick,
#     the final model uses N_MLPS = 5)
# -----------------------------------------------------------------------------
section("3. HYPER-PARAMETER SEARCH FOR THE BETTER MODEL  [3-fold CV on the TRAIN set only]")
print("  the blend weight cannot be read off one validation split: measured on this data, a single split picks")
print("  0.7 while 3-fold CV on the train set (and the paired test runs) prefer the MLP-dominated end.")
search_rows = []
search_folds = stratified_folds(y_tr, SEARCH_FOLDS, SEED)      # indices into the TRAIN arrays
for w in SEARCH_WEIGHTS:
    t0 = time.time()
    fold_scores = []
    for f_tr, f_va in search_folds:
        bm = BlendEnsembleScratch(n_mlps=3, ebm_weight=w, random_state=SEED).fit(
            X_tr[f_tr], y_tr[f_tr], eval_set=[(X_tr[f_va], y_tr[f_va])], feature_names=feature_cols)
        fold_scores.append(roc_auc(y_tr[f_va], bm.predict_proba(X_tr[f_va])[:, 1]))
    search_rows.append({"ebm_weight (MLP gets 1-w)": w, "CV ROC-AUC": float(np.mean(fold_scores)),
                        "CV ROC-AUC SD": float(np.std(fold_scores)), "fit seconds": round(time.time() - t0, 1)})
    print(f"    w={w:<4} CV ROC-AUC {np.mean(fold_scores):.4f} (+-{np.std(fold_scores):.4f})", flush=True)
search = pd.DataFrame(search_rows).set_index("ebm_weight (MLP gets 1-w)")
EBM_WEIGHT = float(search["CV ROC-AUC"].idxmax())
search.to_csv(out("other", "classification_model_search.csv"), index_label="ebm_weight")
print(search.to_string())
print(f"Chosen by {SEARCH_FOLDS}-fold CV on the train set: ebm_weight = {EBM_WEIGHT} (n_mlps = {N_MLPS}) - the EBM is kept in")
print("  for interpretability, but the bagged MLP carries the ranking; the test set is never used to choose it.")
print("  measured (paired over 4 splits, selection on validation/train only): adding Logistic Regression / SVC /")
print("  Gradient Boosting / single-layer perceptron to this blend LOWERS its ROC (-0.0002 to -0.0011), using 9 MLP")
print("  bags instead of 5 changes nothing, and a 250-500 unit MLP is clearly worse (-0.001 to -0.006)")
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(search.index, search["CV ROC-AUC"], "o-", color=plots.RED, label="CV ROC-AUC on the train folds")
ax.fill_between(search.index, search["CV ROC-AUC"] - search["CV ROC-AUC SD"],
                search["CV ROC-AUC"] + search["CV ROC-AUC SD"], color=plots.RED, alpha=0.12)
ax.axvline(EBM_WEIGHT, color=plots.NEUTRAL, linestyle=":", label=f"chosen {EBM_WEIGHT}")
ax.set_xlabel("EBM weight of the blend (MLP bag gets 1 - w)"); ax.set_ylabel("ROC-AUC")
ax.set_title(f"Blend weight chosen by {SEARCH_FOLDS}-fold CV on the train set (never on the test set)"); ax.legend()
plots._save(fig, out("other", "classification_model_search.png"))

# -----------------------------------------------------------------------------
# 4. Train the 15 scratch models
# -----------------------------------------------------------------------------
section("4. TRAINING THE 15 SCRATCH MODELS  [artifact: A. Classification]")
print("  model                        fit | ROC-AUC | PR-AUC | threshold | defaulters caught")
models = make_models(EBM_WEIGHT)
fitted, p_tr, p_va, p_te, rows = {}, {}, {}, {}, []
for name, (factory, use_val) in models.items():
    t0 = time.time()
    m = fit_model(factory, use_val, X_tr, y_tr, X_va, y_va, feature_cols)
    fitted[name] = m
    p_tr[name] = m.predict_proba(X_tr)[:, 1]
    p_va[name] = m.predict_proba(X_va)[:, 1]
    p_te[name] = m.predict_proba(X_te)[:, 1]
    rep = evaluate_model(y_va, p_va[name], y_te, p_te[name], rule=FLAG_RULE, base_rate=float(y_va.mean()))
    rows.append({"Model": name, **rep})
    print(f"  {name:<28} {time.time() - t0:5.1f}s | ROC {rep['test_roc_auc']:.3f} | PR {rep['test_pr_auc']:.3f} | "
          f"thr {rep['chosen_threshold']:.3f} | catches {rep['test_recall']:.1%} of defaulters ({rep['test_fn']:>2} missed)")
bench = pd.DataFrame(rows).set_index("Model")
# sensitivity (= recall) and specificity (= true-negative rate) from the confusion-matrix counts
bench["test_specificity"] = bench["test_tn"] / (bench["test_tn"] + bench["test_fp"])
bench["test_balanced_accuracy"] = (bench["test_recall"] + bench["test_specificity"]) / 2
bench["test_missed_defaulters"] = bench["test_fn"]
bench["train_logloss"] = [logloss(y_tr, p_tr[n]) if n in p_tr else logloss(y_tr, np.full(len(y_tr), y_tr.mean())) for n in bench.index]
bench["val_logloss"] = [logloss(y_va, p_va[n]) if n in p_va else logloss(y_va, np.full(len(y_va), y_tr.mean())) for n in bench.index]

# -----------------------------------------------------------------------------
# 5. Loss (log loss, i.e. how surprised the model is by the real labels)
# -----------------------------------------------------------------------------
section("5. LOSS (log loss)  [artifact: Loss]")
loss_table = bench[["train_logloss", "val_logloss", "test_logloss"]].rename(
    columns={"train_logloss": "Train log loss", "val_logloss": "Validation log loss", "test_logloss": "Test log loss"})
loss_table.to_csv(out("loss", "classification_loss_table.csv"))
print("  log loss (lower is better). NOTE: on 16% positives a model that always answers the base rate "
      "scores a misleadingly low log loss - judge the models on ROC-AUC / PR-AUC / recall instead")
print(loss_table.to_string(float_format=lambda x: f"{x:.4f}"))
print(f"\n  best model log loss = {loss_table['Test log loss'].min():.4f} "
      f"({loss_table['Test log loss'].idxmin()}), worst = {loss_table['Test log loss'].max():.4f} "
      f"({loss_table['Test log loss'].idxmax()})")
fig, ax = plt.subplots(figsize=(12, 4.6))
x = np.arange(len(loss_table)); w = 0.27
for k, (col, alpha_, lab) in enumerate([("Train log loss", 0.35, "train"), ("Validation log loss", 0.65, "validation"),
                                        ("Test log loss", 1.0, "test")]):
    ax.bar(x + (k - 1) * w, loss_table[col], w, label=lab, color=[COLORS[n] for n in loss_table.index], alpha=alpha_)
    for xi, v in zip(x + (k - 1) * w, loss_table[col]):
        ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", fontsize=5.5, rotation=90)
ax.set_xticks(x, loss_table.index, rotation=35, ha="right", fontsize=8); ax.set_ylabel("log loss (lower is better)")
ax.set_title("Log loss on the train / validation / test sets, every model")
ax.legend()
plots._save(fig, out("loss", "classification_train_vs_test_loss.png"))

# training loss curves: models that record one get a curve, the rest are shown as a flat reference line
histories = {n: list(m.train_loss) for n, m in fitted.items() if getattr(m, "train_loss", None)}
no_hist = [n for n in fitted if n not in histories]
print(f"\n  models with a per-iteration training curve: {', '.join(histories)}")
print(f"  models without one (batch/tree models - a single fit, no iterations to plot): {', '.join(no_hist)}")
print(f"  their final train log loss: " + ", ".join(f"{n} {loss_table.loc[n, 'Train log loss']:.4f}" for n in no_hist))
pd.DataFrame([{"model": n, "iteration": i + 1, "train_logloss": v} for n, h in histories.items()
              for i, v in enumerate(h)]).to_csv(out("loss", "classification_loss_curves.csv"), index=False)
pd.DataFrame([{"model": n, "train_logloss_final": loss_table.loc[n, "Train log loss"], "per_iteration_history": False}
              for n in no_hist]).to_csv(out("loss", "classification_loss_curves_no_history.csv"), index=False)

ncol = 4
nrow = (len(fitted) + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 2.9 * nrow)); axes = np.atleast_1d(axes).ravel()
for ax, name in zip(axes, fitted):
    h = histories.get(name)
    if h:
        ax.plot(np.arange(1, len(h) + 1), h, color=COLORS[name])
        ax.plot([len(h)], [h[-1]], "o", color=COLORS[name], markersize=6)
        ax.set_xlabel("iteration"); ax.set_ylabel("train log loss")
        marks = [0, len(h) // 2, len(h) - 1]
        print(f"  {name:<26} " + "  ".join(f"it {m + 1}: {h[m]:.5f}" for m in marks))
    else:
        v = loss_table.loc[name, "Train log loss"]
        ax.axhline(v, color=COLORS[name], linewidth=2)
        ax.plot([0], [v], "o", color=COLORS[name], markersize=6)
        ax.set_xlim(0, 1); ax.set_ylim(max(0.0, v - 0.15), v + 0.15)
        ax.set_xlabel("no per-iteration history"); ax.set_ylabel("train log loss")
        ax.text(0.5, v, f" {v:.4f} (single fit)", va="bottom", ha="center", fontsize=8, color=COLORS[name])
    ax.set_title(name, fontsize=9)
for ax in axes[len(fitted):]:
    ax.axis("off")
fig.suptitle("Training log loss: models with an iteration curve (left to right, same order as the model list) "
             "and batch/tree models as a flat single-fit reference")
plots._save(fig, out("loss", "classification_loss_curves.png"))

# --- the same information as ONE big chart: every model on a single pair of axes ---
knn_like = [n for n in histories if n == "k-NN"]          # its history is a k-scan, not a training curve
curve_models = [n for n in histories if n not in knn_like]
flat = [n for n in fitted if n not in histories]


def spread_labels(ys, gap):
    """Push label positions apart so the names never overlap (keeps the original order)."""
    out = []
    for i, y in enumerate(ys):
        out.append(y if i == 0 else max(y, out[i - 1] + gap))
    return out


fig, ax = plt.subplots(figsize=(14.5, 8.0))
# x = each model's own progress in percent: iteration counts here run from 3 to 1500, and plotting the raw
# iteration numbers squeezes every curve into the far left (a 1500-step SVC flattens the whole chart)
ends = sorted((histories[n][-1], n, len(histories[n])) for n in curve_models)
lab_y = spread_labels([e[0] for e in ends], 0.0105)
for (y_end, name, n_steps), y_lab in zip(ends, lab_y):
    ax.plot(np.arange(1, n_steps + 1) / n_steps * 100, histories[name], color=COLORS[name], linewidth=2.3,
            solid_capstyle="round", alpha=0.95, zorder=3)
    ax.plot([100], [y_end], "o", color=COLORS[name], markersize=6, markeredgecolor="white",
            markeredgewidth=1.2, zorder=6)
    ax.annotate(f"{name}  {y_end:.3f}  ({n_steps:,} steps)", xy=(100, y_end), xytext=(104, y_lab),
                textcoords="data", va="center", fontsize=9, color=COLORS[name], fontweight="bold", zorder=7)
# batch / tree models have no iteration curve: one diamond each, spread out on the left
flat_pts = sorted((float(loss_table.loc[n, "Train log loss"]), n) for n in flat)
flat_lab = spread_labels([v for v, _ in flat_pts], 0.0135)
for (v, name), y_lab in zip(flat_pts, flat_lab):
    ax.plot([1], [v], "D", color=COLORS[name], markersize=8, markeredgecolor="white",
            markeredgewidth=1.2, zorder=6)
    ax.annotate(f"{name}  {v:.3f}  (single fit, no curve)", xy=(1, v), xytext=(3.5, y_lab), textcoords="data",
                va="center", fontsize=8.5, color=COLORS[name])
ax.axhline(np.log(2), color=plots.NEUTRAL, linestyle="--", linewidth=1.3, zorder=1,
           label="answering 50/50 for everyone = log(2) = 0.693")
ax.set_xlabel("training progress of that model  (% of its own iterations - models have very different step counts)")
ax.set_ylabel("training log loss   (lower is better)")
ax.set_xlim(-2, 132)
lo = min([e[0] for e in ends] + [v for v, _ in flat_pts] + [np.log(2)])
hi = max([e[0] for e in ends] + [v for v, _ in flat_pts] + [np.log(2)])
ax.set_ylim(lo - 0.02, hi + 0.03)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.45)
ax.legend(fontsize=8.5, loc="lower center", framealpha=0.92, borderpad=0.7)
ax.set_title("Training loss of every model on one chart  --  each name sits next to its own curve (lower = better)",
             fontsize=12.5, fontweight="bold", pad=16)
ax.text(0.5, 1.012, f"{len(curve_models)} models drawn as curves, {len(flat)} batch/tree models as a diamond, "
                    f"k-NN left out (its history is a k-scan, see the panel chart); "
                    f"models are shown at equal width so a 3-step and a 1,500-step model can be compared",
        transform=ax.transAxes, ha="center", fontsize=8.5, color=plots.NEUTRAL)
plots._save(fig, out("loss", "classification_loss_curves_all_models.png"))

# -----------------------------------------------------------------------------
# 6. Catching defaulters: the flagging threshold and how many defaulters a top-x% shortlist catches
# -----------------------------------------------------------------------------
section("6. CATCHING DEFAULTERS  [artifact: analysis & discussion]  (threshold rules + recall@top-k)")
rule_rows = []
LADDER = ["0.5 (plain)", "f1", "f1.5", "f2", "youden", "precision>=1.5x base"]
ladder = {r: {"recall": [], "precision": [], "flagged": []} for r in LADDER}
for name in fitted:
    thr_used = float(bench.loc[name, "chosen_threshold"])       # the headline rule, chosen on validation
    flag_used = float((p_te[name] >= thr_used).mean())
    rule_rows.append({
        "Model": name,
        f"thr {FLAG_RULE} (in use)": thr_used,
        "thr F1": bench.loc[name, "thr_f1_rule"], "thr Youden": bench.loc[name, "thr_youden_rule"],
        f"recall @{FLAG_RULE}": bench.loc[name, "test_recall"],
        "recall @F1": bench.loc[name, "test_recall@f1_rule"],
        "recall @0.5": bench.loc[name, "test_recall@0.5"],
        f"precision @{FLAG_RULE}": bench.loc[name, "test_precision"],
        "precision @0.5": bench.loc[name, "test_precision@0.5"],
        "applicants flagged": flag_used,
        "enrichment of the flagged group": bench.loc[name, "test_precision"] / float(y_te.mean()),
        "defaulters missed": bench.loc[name, "test_fn"]})
    for r in LADDER:                                            # the rule ladder: what each rule would give
        t = 0.5 if r.startswith("0.5") else choose_threshold(y_va, p_va[name], rule=r, base_rate=float(y_va.mean()))[0]
        ladder[r]["recall"].append(recall(y_te, p_te[name], t))
        ladder[r]["precision"].append(precision(y_te, p_te[name], t))
        ladder[r]["flagged"].append(float((p_te[name] >= t).mean()))
rules = pd.DataFrame(rule_rows).set_index("Model")
rules.to_csv(out("other", "classification_threshold_rules.csv"), index_label="Model")
best_recall = bench["test_recall"].idxmax(); best_roc = bench["test_roc_auc"].idxmax()

print(f"  rule in use = '{FLAG_RULE}' (switch it with the one-word constant FLAG_RULE at the top of the file)")
print(f"  flagged = share of the {len(y_te)} test applicants that gets sent for review; "
      f"enrichment = how much more likely a flagged applicant is to default than a random one (1.00 = useless)")
print(rules.rename(columns={f"thr {FLAG_RULE} (in use)": "threshold", f"recall @{FLAG_RULE}": "recall",
                            f"precision @{FLAG_RULE}": "precision", "recall @F1": "recall@F1", "recall @0.5": "recall@0.5",
                            "precision @0.5": "prec@0.5", "applicants flagged": "flagged",
                            "enrichment of the flagged group": "enrichment",
                            "defaulters missed": "missed"})[["threshold", "recall", "precision", "flagged", "enrichment",
                                                             "recall@F1", "recall@0.5", "prec@0.5", "missed"]].to_string(
    float_format=lambda x: f"{x:.3f}"))

ladder_tbl = pd.DataFrame({r: {"recall (defaulters caught)": np.mean(v["recall"]), "precision": np.mean(v["precision"]),
                               "applicants flagged": np.mean(v["flagged"]),
                               "enrichment vs random": np.mean(v["precision"]) / float(y_te.mean()),
                               "defaulters missed (of 305)": (1 - np.mean(v["recall"])) * float(y_te.sum())}
                           for r, v in ladder.items()}).T
ladder_tbl.to_csv(out("other", "classification_threshold_ladder.csv"), index_label="rule")
print("\n  RULE LADDER (average over the 15 models, every rule tuned on validation then measured on test):")
print(ladder_tbl.to_string(float_format=lambda x: f"{x:.3f}"))
print(f"  -> the rule in use, '{FLAG_RULE}', flags {ladder_tbl.loc[FLAG_RULE, 'applicants flagged']:.1%} of the applicants "
      f"and catches {ladder_tbl.loc[FLAG_RULE, 'recall (defaulters caught)']:.1%} of the defaulters "
      f"({ladder_tbl.loc[FLAG_RULE, 'enrichment vs random']:.2f}x enriched)")
print(f"  -> a plain 0.5 flags only {ladder_tbl.loc['0.5 (plain)', 'applicants flagged']:.1%} but catches "
      f"{ladder_tbl.loc['0.5 (plain)', 'recall (defaulters caught)']:.1%}; the catch-max rule f2 catches "
      f"{ladder_tbl.loc['f2', 'recall (defaulters caught)']:.1%} but flags {ladder_tbl.loc['f2', 'applicants flagged']:.1%}")
print(f"Most defaulters caught: {best_recall} ({bench.loc[best_recall, 'test_recall']:.3f} of all defaulters, "
      f"{int(bench.loc[best_recall, 'test_fn'])} missed); best ranking: {best_roc} (ROC-AUC {bench.loc[best_roc, 'test_roc_auc']:.4f})")

fig, axes = plt.subplots(1, 3, figsize=(19, 5.0))
o = rules.index[np.argsort(rules["recall @0.5"].values)]
xr = np.arange(len(o))
for k, (col, lab) in enumerate([(f"recall @{FLAG_RULE}", f"in use: {FLAG_RULE} (catches defaulters)"),
                                ("recall @F1", "F1 (balanced)"), ("recall @0.5", "plain 0.5")]):
    if col not in rules:
        continue
    axes[0].barh(xr + (k - 1) * 0.27, rules.loc[o, col], 0.27, label=lab,
                 color=[COLORS[n] for n in o], alpha=[1.0, 0.62, 0.35][k])
axes[0].set_yticks(xr, o, fontsize=8); axes[0].set_xlabel("share of defaulters caught on the test set")
axes[0].set_title("Recall on the defaulters: which rule you pick matters"); axes[0].legend(fontsize=8); axes[0].grid(axis="y", visible=False)
axes[1].barh(xr, rules.loc[o, "applicants flagged"], 0.6, color=[COLORS[n] for n in o])
axes[1].set_yticks(xr, o, fontsize=8); axes[1].set_xlabel("share of applicants flagged for review")
axes[1].axvline(float(y_te.mean()), color=plots.NEUTRAL, linestyle="--", label=f"base rate {y_te.mean():.1%}")
axes[1].set_title("How many applicants get flagged"); axes[1].legend(fontsize=8); axes[1].grid(axis="y", visible=False)
axes[2].barh(xr, rules.loc[o, "enrichment of the flagged group"], 0.6, color=[COLORS[n] for n in o])
axes[2].set_yticks(xr, o, fontsize=8); axes[2].set_xlabel("enrichment vs a random applicant")
axes[2].axvline(1.0, color=plots.NEUTRAL, linestyle="--", label="1.00 = no better than random")
axes[2].set_title("How much better the flagged group is"); axes[2].legend(fontsize=8); axes[2].grid(axis="y", visible=False)
fig.suptitle(f"Flagging rule '{FLAG_RULE}': recall, how many applicants it flags, and how enriched that group is "
             f"(thresholds tuned on validation, measured on test)")
plots._save(fig, out("other", "classification_threshold_rules.png"))

# recall@top-k: rank the applicants by risk, flag the riskiest x%, count the defaulters inside
topk_rows = []
for name in fitted:
    for r in topk_table(y_te, p_te[name]):
        topk_rows.append({"Model": name, **r})
topk = pd.DataFrame(topk_rows)
topk.to_csv(out("other", "classification_recall_at_topk.csv"), index=False)
pivot = topk.pivot(index="flagged share", columns="Model", values="recall (defaulters caught / all)")
recap = (pivot.T * 100).round(1)                       # models as ROWS: far easier to read than 15 columns
recap.columns = [f"top {int(round(c * 100))}%" for c in recap.columns]
print("\nRecall = share of ALL defaulters caught, if the riskiest x% of applicants is flagged first")
print(recap.to_string(float_format=lambda x: f"{x:5.1f}"))
lift = topk.pivot(index="flagged share", columns="Model", values="lift vs flagging at random").T.reindex(recap.index)
lift.columns = [f"top {int(round(c * 100))}%" for c in lift.columns]
print("\nLift = how many times better than flagging applicants at random (1.00 = no better than guessing)")
print(lift.to_string(float_format=lambda x: f"{x:5.2f}"))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(16.5, 5.6))
for name in fitted:
    sub_r = topk[topk["Model"] == name]
    a1.plot(sub_r["flagged share"], sub_r["recall (defaulters caught / all)"], "o-", color=COLORS[name], label=name, linewidth=1.4, markersize=4)
    a2.plot(sub_r["flagged share"], sub_r["lift vs flagging at random"], "o-", color=COLORS[name], label=name, linewidth=1.4, markersize=4)
order_by_roc = bench["test_roc_auc"].sort_values(ascending=False).index
a1.plot(TOPK, [f for f in TOPK], "--", color=plots.NEUTRAL, label="flag at random")
a1.set_xlabel("share of applicants flagged (highest risk first)"); a1.set_ylabel("share of all defaulters caught")
a1.set_title("Cumulative gains: how many defaulters the shortlist catches"); a1.legend(fontsize=7, ncol=2)
a2.axhline(1.0, color=plots.NEUTRAL, linestyle="--"); a2.set_xlabel("share of applicants flagged"); a2.set_ylabel("lift vs random")
a2.set_title("Lift of the shortlist over flagging at random"); a2.legend(fontsize=7, ncol=2)
plots._save(fig, out("other", "classification_recall_at_topk.png"))

best_top20 = pivot.loc[0.20].idxmax()
print(f"\nFlagging the riskiest 20% of applicants: best model = {best_top20} "
      f"({pivot.loc[0.20, best_top20]:.1%} of all defaulters caught, "
      f"precision inside the group {topk[(topk['Model'] == best_top20) & (topk['flagged share'] == 0.20)]['precision inside the flagged group'].iloc[0]:.1%})")

# -----------------------------------------------------------------------------
# 7. Confusion matrix, accuracy, precision / recall / F1 (threshold chosen on validation)
# -----------------------------------------------------------------------------
section(f"7. CONFUSION MATRIX / ACCURACY / PRECISION-RECALL-F1  ({FLAG_RULE} threshold chosen on validation)")
print(bench[["chosen_threshold", "test_accuracy", "test_precision", "test_recall", "test_specificity", "test_f1",
             "test_roc_auc", "test_pr_auc", "test_fn"]].to_string())
bench[[c for c in bench.columns if c.startswith(("test_", "val_")) or c in ("chosen_threshold", "threshold_rule")]].to_csv(
    out("other", "classification_benchmark_all_metrics.csv"))
bench[[c for c in bench.columns if c.startswith("test_")]].to_csv(
    out("other", "classification_benchmark_test_metrics_only.csv"))
bench[["test_accuracy", "test_accuracy@0.5"]].to_csv(out("acc", "classification_accuracy.csv"))
bench[["test_precision", "test_recall", "test_f1"]].to_csv(out("prf", "classification_precision_recall_f1.csv"))
bench[["test_recall", "test_specificity", "test_balanced_accuracy"]].rename(columns={"test_recall": "sensitivity (recall)", "test_specificity": "specificity (TNR)", "test_balanced_accuracy": "balanced accuracy"}).to_csv(out("prf", "classification_sensitivity_specificity.csv"))
bench[["test_roc_auc", "test_pr_auc"]].to_csv(out("roc", "classification_roc_auc.csv"))

def bar(col, title, topic, fname, ylabel, ref=None, ref_label=""):
    """Horizontal bars for every model, with an optional reference line (e.g. what guessing scores)."""
    fig, ax = plt.subplots(figsize=(12, 5.0))
    order = bench[col].sort_values().index
    ax.barh(order, bench.loc[order, col], color=[COLORS[n] for n in order], height=0.62)
    if ref is not None:
        ax.axvline(ref, color=plots.NEUTRAL, linestyle="--", label=f"{ref_label} = {ref:.3f}")
    ax.set_title(title); ax.set_xlabel(ylabel); ax.grid(axis="y", visible=False); ax.legend(fontsize=8)
    for name, v in bench.loc[order, col].items():
        ax.text(v, name, f" {v:.3f}", va="center", fontsize=7)
    ax.set_xlim(0, bench[col].max() * 1.16)
    plots._save(fig, out(topic, fname))

bar("test_accuracy", "Test accuracy (at the validation-chosen threshold)", "acc", "classification_accuracy.png", "accuracy")
bar("test_roc_auc", "Test ROC-AUC, every model", "roc", "classification_roc_auc_bars.png", "ROC-AUC", ref=0.5, ref_label="coin toss")
bar("test_pr_auc", "Test PR-AUC, every model", "roc", "classification_pr_auc_bars.png", "PR-AUC",
    ref=float(y_te.mean()), ref_label="random (base rate)")
bar("test_balanced_accuracy", "Test balanced accuracy (sensitivity + specificity) / 2", "acc", "classification_balanced_accuracy.png", "balanced accuracy")
bar("test_recall", "Defaulters caught (recall) on the test set", "prf", "classification_recall_bars.png", "recall")

fig, ax = plt.subplots(figsize=(12, 4.4))
w = 0.26
o = bench["test_f1"].sort_values().index
for k, (col, lab) in enumerate([("test_precision", "precision"), ("test_recall", "recall (defaulters caught)"), ("test_f1", "F1")]):
    ax.barh(np.arange(len(o)) + (k - 1) * w, bench.loc[o, col], w, label=lab)
ax.set_yticks(np.arange(len(o)), o, fontsize=8); ax.legend(); ax.grid(axis="y", visible=False)
ax.set_title(f"Precision / recall / F1 on the test set, every model ({FLAG_RULE} threshold)")
plots._save(fig, out("prf", "classification_precision_recall_f1.png"))

fig, ax = plt.subplots(figsize=(12, 4.4))
o = bench["test_recall"].sort_values().index
for k, (col, lab) in enumerate([("test_recall", "sensitivity (defaulters caught)"), ("test_specificity", "specificity (good payers kept)")]):
    ax.barh(np.arange(len(o)) + (k - 0.5) * 0.36, bench.loc[o, col], 0.36, label=lab)
ax.set_yticks(np.arange(len(o)), o, fontsize=8); ax.legend(); ax.grid(axis="y", visible=False)
ax.set_title("Sensitivity vs specificity on the test set, every model (validation-chosen threshold)")
plots._save(fig, out("prf", "classification_sensitivity_specificity.png"))

ncol = 4; nrow = (len(fitted) + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow)); axes = np.atleast_1d(axes).ravel()
for ax in axes[len(fitted):]:
    ax.axis("off")
for ax, name in zip(axes, fitted):
    cm = confusion_matrix(y_te, p_te[name], bench.loc[name, "chosen_threshold"])
    ax.imshow(cm, cmap="Blues"); ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="black", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"], fontsize=7); ax.set_yticklabels(["true 0", "true 1"], fontsize=7)
    ax.set_title(f"{name}\nthr {bench.loc[name, 'chosen_threshold']:.3f} | missed defaulters {bench.loc[name, 'test_fn']}", fontsize=8)
plots._save(fig, out("cm", "classification_confusion_matrices.png"))

# the same grid as a table: how much of each true class is caught
cm_rows = []
for name in fitted:
    tn, fp, fn, tp = confusion_matrix(y_te, p_te[name], bench.loc[name, "chosen_threshold"]).ravel()
    cm_rows.append({"Model": name, "threshold": bench.loc[name, "chosen_threshold"], "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                    "defaulters caught (recall)": tp / (tp + fn), "good payers kept (specificity)": tn / (tn + fp)})
pd.DataFrame(cm_rows).to_csv(out("cm", "classification_confusion_matrix_counts.csv"), index=False)
print("\n" + pd.DataFrame(cm_rows).set_index("Model").to_string())

# -----------------------------------------------------------------------------
# 8. ROC-AUC / PR-AUC: the curves themselves
# -----------------------------------------------------------------------------
section("8. ROC-AUC AND PR-AUC  [artifact: ROC-AUC]")
print(bench[["test_roc_auc", "test_pr_auc"]].sort_values("test_roc_auc", ascending=False).to_string())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.5, 5.4))
for name in fitted:
    fpr, tpr, _ = roc_curve_scratch(y_te, p_te[name])
    a1.plot(fpr, tpr, color=COLORS[name], label=f"{name} ({bench.loc[name, 'test_roc_auc']:.3f})", linewidth=1.4)
    pr, rc, _ = precision_recall_curve_scratch(y_te, p_te[name])
    a2.plot(rc, pr, color=COLORS[name], label=f"{name} ({bench.loc[name, 'test_pr_auc']:.3f})", linewidth=1.4)
a1.plot([0, 1], [0, 1], color=plots.NEUTRAL, linestyle="--", label="random (0.500)")
a1.set_xlabel("false positive rate"); a1.set_ylabel("true positive rate"); a1.set_title("ROC curve, every model"); a1.legend(fontsize=6.5, ncol=2)
a2.axhline(y_te.mean(), color=plots.NEUTRAL, linestyle="--", label=f"base rate ({y_te.mean():.3f})")
a2.set_xlabel("recall (defaulters caught)"); a2.set_ylabel("precision"); a2.set_title("Precision-recall curve, every model"); a2.legend(fontsize=6.5, ncol=2)
plots._save(fig, out("roc", "classification_roc_and_pr_curves.png"))

# -----------------------------------------------------------------------------
# 9. Performance curve: validation loss / PR-AUC while a model is being trained
#    every model that exposes staged_predict_proba() gets its own curve
# -----------------------------------------------------------------------------
section("9. PERFORMANCE CURVE OF EVERY STAGED MODEL  [artifact: Performance curve]")
staged_models = [n for n, m in fitted.items() if hasattr(m, "staged_predict_proba")]
print(f"  models with staged predictions (one curve each): {', '.join(staged_models)}")
not_staged = [n for n in fitted if n not in staged_models]
print(f"  models without staged predictions (no iteration structure to plot): {', '.join(not_staged)}")
perf_rows = []
staged_ll = {}
for name in staged_models:
    st = _staged_pos(fitted[name], X_va)
    ll = [logloss(y_va, p) for p in st]
    pr_ = [average_precision(y_va, p) for p in st]
    staged_ll[name] = ll
    for i, (a, b) in enumerate(zip(ll, pr_)):
        perf_rows.append({"Model": name, "iteration": i + 1, "val_logloss": a, "val_pr_auc": b})
    print(f"  {name:<26} best validation log loss {min(ll):.4f} at iteration {int(np.argmin(ll)) + 1} of {len(ll)}; "
          f"validation PR-AUC peaks {max(pr_):.4f} at {int(np.argmax(pr_)) + 1}")
pd.DataFrame(perf_rows).to_csv(out("perf", "classification_performance_curves.csv"), index=False)

fig, ax = plt.subplots(figsize=(10, 4.6))
for name in staged_models:
    ll = staged_ll[name]
    ax.plot(np.arange(1, len(ll) + 1), ll, color=COLORS[name], label=f"{name} (min {min(ll):.4f})", linewidth=1.5)
ax.set_xlabel("iteration (boosting round / gradient-descent step)"); ax.set_ylabel("validation log loss")
ax.set_title("Validation loss while training -- every model that can be evaluated stage by stage"); ax.legend(fontsize=8)
plots._save(fig, out("perf", "classification_validation_curves.png"))

fig, ax = plt.subplots(figsize=(9, 4.4))
for name in staged_models:
    pr_ = [r["val_pr_auc"] for r in perf_rows if r["Model"] == name]
    ax.plot(np.arange(1, len(pr_) + 1), pr_, color=COLORS[name], label=name, linewidth=1.5)
ax.axhline(y_va.mean(), color=plots.NEUTRAL, linestyle="--", label=f"base rate ({y_va.mean():.3f})")
ax.set_xlabel("iteration"); ax.set_ylabel("validation PR-AUC")
ax.set_title("Validation PR-AUC while training (the metric that matters for the 16% positive class)"); ax.legend(fontsize=8)
plots._save(fig, out("perf", "classification_validation_pr_auc_curves.png"))

# the best boosting model in detail: loss and PR-AUC on one chart, with the early-stopping point
boost_names = [n for n in ("Gradient Boosting", "XGBoost") if n in fitted]
boost_name = max(boost_names, key=lambda n: average_precision(y_va, p_va[n]))
st = _staged_pos(fitted[boost_name], X_va)
ll = [logloss(y_va, p) for p in st]
pr_ = [average_precision(y_va, p) for p in st]
fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.plot(np.arange(1, len(st) + 1), ll, color=plots.BLUE, label="validation log loss")
ax.set_xlabel(f"number of trees ({boost_name})"); ax.set_ylabel("validation log loss", color=plots.BLUE)
ax2 = ax.twinx()
ax2.plot(np.arange(1, len(st) + 1), pr_, color=plots.RED, label="validation PR-AUC")
ax2.set_ylabel("validation PR-AUC", color=plots.RED); ax2.grid(False)
ax.axvline(int(np.argmin(ll)) + 1, color=plots.NEUTRAL, linestyle=":", label=f"best log loss at tree {int(np.argmin(ll)) + 1}")
ax.set_title(f"{boost_name}: validation performance vs. number of trees"); ax.legend(loc="lower left", fontsize=8)
plots._save(fig, out("perf", f"classification_{boost_name.split()[0].lower()}_performance_curve.png"))

# -----------------------------------------------------------------------------
# 10. Extra check: the scratch models vs their ready-to-use scikit-learn counterparts, every model
# -----------------------------------------------------------------------------
section("10. VERIFICATION: SCRATCH vs SCIKIT-LEARN, EVERY MODEL  [artifact: ready-to-use function check]")
w_tr = np.where(y_tr == 1, np.sqrt((len(y_tr) - y_tr.sum()) / y_tr.sum()), 1.0)      # project imbalance weight
ver = []


def add_check(label, diff, auc_scratch, auc_sklearn, meaning):
    ver.append((label, diff, auc_scratch, auc_sklearn, meaning))
    print(f"  [{len(ver):>2}] {label}")


try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import CategoricalNB, GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler as _SK
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier

    # --- 1. Logistic regression ---
    sc = _SK().fit(X_tr)
    lr_sk = LogisticRegression(max_iter=5000, tol=1e-10).fit(sc.transform(X_tr), y_tr, sample_weight=w_tr)
    add_check("Logistic Regression vs sklearn LogisticRegression",
              float(np.abs(fitted["Logistic Regression"].coef_ - lr_sk.coef_[0]).max()),
              roc_auc(y_te, p_te["Logistic Regression"]), roc_auc(y_te, lr_sk.predict_proba(sc.transform(X_te))[:, 1]), "max |dcoef|")

    # --- 2. Decision tree: same hyper-parameters -> the predictions should be identical ---
    dt_sk = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=SEED).fit(X_tr, y_tr)
    same = float(np.mean(dt_sk.predict(X_te) == fitted["Decision Tree"].predict(X_te)))
    add_check("Decision Tree vs sklearn DecisionTreeClassifier (identical predictions)",
              1.0 - same, roc_auc(y_te, p_te["Decision Tree"]), roc_auc(y_te, dt_sk.predict_proba(X_te)[:, 1]), "1 - share identical")

    # --- 3. Random forest and 4. Gradient boosting ---
    rf_sk = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
    add_check("Random Forest vs sklearn RandomForestClassifier",
              float(np.abs(p_te["Random Forest"] - rf_sk.predict_proba(X_te)[:, 1]).max()),
              roc_auc(y_te, p_te["Random Forest"]), roc_auc(y_te, rf_sk.predict_proba(X_te)[:, 1]), "max |dP|")
    gb_sk = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED).fit(X_tr, y_tr)
    add_check("Gradient Boosting vs sklearn GradientBoostingClassifier",
              float(np.abs(p_te["Gradient Boosting"] - gb_sk.predict_proba(X_te)[:, 1]).max()),
              roc_auc(y_te, p_te["Gradient Boosting"]), roc_auc(y_te, gb_sk.predict_proba(X_te)[:, 1]), "max |dP|")
    try:
        from xgboost import XGBClassifier
        xg_sk = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                              random_state=SEED, eval_metric="logloss").fit(X_tr, y_tr)
        add_check("XGBoost vs real xgboost (same settings)",
                  float(np.abs(p_te["XGBoost"] - xg_sk.predict_proba(X_te)[:, 1]).max()),
                  roc_auc(y_te, p_te["XGBoost"]), roc_auc(y_te, xg_sk.predict_proba(X_te)[:, 1]), "max |dP|")
    except Exception as e:
        print(f"  (skipped real xgboost: {type(e).__name__})")

    # --- 5. SVC (linear) and 6. k-NN ---
    # libsvm's SVC(kernel="linear") needs many minutes on 5,746 x 27 with balanced weights, so the
    # counterpart used here is LinearSVC (liblinear, the same linear hinge + balanced weights) which is seconds.
    sv_sk = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED).fit(X_tr, y_tr)
    add_check("SVC (linear) vs sklearn LinearSVC (same hinge + balanced weights, liblinear solver)",
              float(np.mean(sv_sk.predict(X_te) == fitted["SVC (linear)"].predict(X_te))),
              roc_auc(y_te, p_te["SVC (linear)"]), roc_auc(y_te, sv_sk.decision_function(X_te)), "share of same flags")
    knn_k = int(getattr(fitted["k-NN"], "n_neighbors_", 25))
    print(f"  k-NN tuned its k on the validation set: k = {knn_k} "
          f"(grid scan, best validation {getattr(fitted['k-NN'], 'k_selection_', {}).get('metric', 'auc')})")
    knn_sk = KNeighborsClassifier(n_neighbors=knn_k, weights="distance").fit(sc.transform(X_tr), y_tr)
    add_check(f"k-NN vs sklearn KNeighborsClassifier (k={knn_k}, distance weights)",
              float(np.abs(p_te["k-NN"] - knn_sk.predict_proba(sc.transform(X_te))[:, 1]).max()),
              roc_auc(y_te, p_te["k-NN"]), roc_auc(y_te, knn_sk.predict_proba(sc.transform(X_te))[:, 1]), "max |dP|")

    # --- 7. Naive Bayes (both flavours) ---
    g_s = NaiveBayesScratch(kind="gaussian").fit(X_tr, y_tr, feature_names=feature_cols)
    g_k = GaussianNB().fit(X_tr, y_tr)
    add_check("Naive Bayes (Gaussian) vs GaussianNB",
              float(np.abs(g_s.predict_proba(X_te)[:, 1] - g_k.predict_proba(X_te)[:, 1]).max()),
              roc_auc(y_te, g_s.predict_proba(X_te)[:, 1]), roc_auc(y_te, g_k.predict_proba(X_te)[:, 1]), "max |dP|")
    nbc = fitted["Naive Bayes"]
    binned = lambda A: np.column_stack([np.searchsorted(nbc.edges_[j], A[:, j], side="right") for j in range(A.shape[1])])
    c_k = CategoricalNB(alpha=1.0, min_categories=[len(e) + 1 for e in nbc.edges_]).fit(binned(X_tr), y_tr)
    add_check("Naive Bayes (categorical) vs CategoricalNB",
              float(np.abs(p_te["Naive Bayes"] - c_k.predict_proba(binned(X_te))[:, 1]).max()),
              roc_auc(y_te, p_te["Naive Bayes"]), roc_auc(y_te, c_k.predict_proba(binned(X_te))[:, 1]), "max |dP|")

    # --- 8. perceptrons, 9. MLP, 10. EBM ---
    p_sk = SGDClassifier(loss="perceptron", penalty=None, learning_rate="constant", eta0=0.1, average=True, max_iter=40, tol=None,
                         class_weight={0: 1, 1: float(w_tr.max())}, random_state=SEED).fit(sc.transform(X_tr), y_tr)
    add_check("Averaged perceptron vs SGDClassifier(perceptron, average)", float("nan"),
              roc_auc(y_te, fitted["Perceptron (averaged)"].decision_function(X_te)), roc_auc(y_te, p_sk.decision_function(sc.transform(X_te))), "AUC only")
    l2 = 1e-3
    slp_full = SingleLayerPerceptronScratch(learning_rate=0.05, n_epochs=400, l2=l2, lr_decay=0.02, random_state=SEED).fit(X_tr, y_tr, early_stopping_rounds=0)
    lr_same = LogisticRegression(C=1 / (l2 * w_tr.sum()), max_iter=5000, tol=1e-10).fit(sc.transform(X_tr), y_tr, sample_weight=w_tr)
    add_check("Single-layer perceptron vs LogisticRegression (same loss)",
              float(np.abs(slp_full.coef_ - lr_same.coef_[0]).max()),
              roc_auc(y_te, slp_full.predict_proba(X_te)[:, 1]), roc_auc(y_te, lr_same.predict_proba(sc.transform(X_te))[:, 1]), "max |dcoef|")
    mlp_sk = MLPClassifier((64, 32), alpha=1e-2, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, max_iter=300, random_state=SEED)
    try:
        mlp_sk.fit(sc.transform(X_tr), y_tr, sample_weight=w_tr)
    except TypeError:
        mlp_sk.fit(sc.transform(X_tr), y_tr)
    add_check("MLP vs sklearn MLPClassifier (same shape)", float("nan"),
              roc_auc(y_te, p_te["MLP"]), roc_auc(y_te, mlp_sk.predict_proba(sc.transform(X_te))[:, 1]), "AUC only")
    add_check("MLP back-propagation vs numerical gradient", MLPClassifierScratch(hidden_layers=(16, 8)).gradient_check(X_tr, y_tr, n_checks=30),
              float("nan"), float("nan"), "max relative error")
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
        ref = ExplainableBoostingClassifier(interactions=0, random_state=SEED, outer_bags=4, learning_rate=0.02, max_bins=16).fit(X_tr, y_tr, sample_weight=w_tr)
        pr_ = ref.predict_proba(X_te)[:, 1]
        add_check("EBM vs interpret.ExplainableBoostingClassifier",
                  float(np.corrcoef(p_te["EBM (additive)"], pr_)[0, 1]), roc_auc(y_te, p_te["EBM (additive)"]), roc_auc(y_te, pr_), "probability correlation")
    except Exception:
        pass

    # --- 11. Stacking ---
    try:
        st_sk = StackingClassifier([("lr", LogisticRegression(max_iter=2000)), ("nb", GaussianNB()),
                                    ("knn", KNeighborsClassifier(25))], final_estimator=LogisticRegression(), cv=3, n_jobs=-1).fit(sc.transform(X_tr), y_tr)
        add_check("Stacking vs sklearn StackingClassifier", float("nan"),
                  roc_auc(y_te, p_te["Stacking (5 bases)"]), roc_auc(y_te, st_sk.predict_proba(sc.transform(X_te))[:, 1]), "AUC only")
    except Exception as e:
        print(f"  (skipped sklearn StackingClassifier: {type(e).__name__})")
except ImportError:
    print("  scikit-learn is not installed: skipping the scratch-vs-sklearn checks (pip install scikit-learn)")
if ver:
    vt = pd.DataFrame(ver, columns=["check", "difference", "AUC scratch", "AUC sklearn", "the difference is"]).set_index("check")
    vt.to_csv(out("other", "classification_scratch_vs_sklearn.csv"))
    print("  every scratch model next to its scikit-learn counterpart (NaN = that check compares AUC only)")
    print(vt[["AUC scratch", "AUC sklearn", "difference", "the difference is"]].round(4).to_string())
    pairs = vt.dropna(subset=["AUC scratch", "AUC sklearn"])
    fig, ax = plt.subplots(figsize=(13, 4.6))
    xs = np.arange(len(pairs))
    ax.barh(xs - 0.19, pairs["AUC scratch"], 0.38, label="scratch (ours)", color=plots.BLUE)
    ax.barh(xs + 0.19, pairs["AUC sklearn"], 0.38, label="scikit-learn", color=plots.NEUTRAL)
    lo = max(0.4, float(pairs[["AUC scratch", "AUC sklearn"]].min().min()) - 0.04)
    ax.set_yticks(xs, [c.split(" vs ")[0] for c in pairs.index], fontsize=7.5)
    ax.set_xlim(lo, min(1.0, float(pairs[["AUC scratch", "AUC sklearn"]].max().max()) + 0.04))
    ax.grid(axis="y", visible=False); ax.legend(); ax.set_xlabel("test ROC-AUC")
    ax.set_title("Scratch models vs their scikit-learn counterparts, every model"); ax.legend(loc="lower right")
    plots._save(fig, out("other", "classification_scratch_vs_sklearn.png"))

# -----------------------------------------------------------------------------
# 11. 5-fold cross-validation (stratified folds)
# -----------------------------------------------------------------------------
section(f"11. {CV_FOLDS}-FOLD CROSS-VALIDATION  [artifact: batch runs]  (mean +/- std of ROC-AUC / PR-AUC / recall)")
folds = stratified_folds(y, CV_FOLDS, SEED)
print("  positive rate per fold: " + ", ".join(f"{y[te].mean():.1%}" for _, te in folds))
cv_scores = {n: {"roc": [], "pr": [], "rec": []} for n in models}
for k, (tr, te) in enumerate(folds, 1):
    cut = int(len(tr) * 0.8)                      # inner validation slice for early stopping
    rk = np.random.RandomState(SEED + k)          # ... stratified by hand, so both slices keep the positive rate
    pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
    rk.shuffle(pos); rk.shuffle(neg)
    n_va = len(tr) - cut
    n_pos_va = max(1, int(round(n_va * len(pos) / len(tr))))
    va_idx = np.concatenate([pos[:n_pos_va], neg[:n_va - n_pos_va]])
    tr_idx = np.setdiff1d(tr, va_idx)
    for name, (factory, use_val) in models.items():
        m = fit_model(factory, use_val, X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], feature_cols)
        p = m.predict_proba(X[te])[:, 1]
        thr, _ = choose_threshold(y[va_idx], m.predict_proba(X[va_idx])[:, 1], rule=FLAG_RULE, base_rate=float(y[va_idx].mean()))
        cv_scores[name]["roc"].append(roc_auc(y[te], p))
        cv_scores[name]["pr"].append(average_precision(y[te], p))
        cv_scores[name]["rec"].append(recall(y[te], p, thr))
    print(f"  fold {k}/{CV_FOLDS} done", flush=True)
cv = pd.DataFrame({n: {"CV ROC-AUC mean": np.mean(v["roc"]), "CV ROC-AUC std": np.std(v["roc"]),
                       "CV PR-AUC mean": np.mean(v["pr"]), "CV PR-AUC std": np.std(v["pr"]),
                       f"CV recall@{FLAG_RULE} mean": np.mean(v["rec"]),
                       f"CV recall@{FLAG_RULE} std": np.std(v["rec"])} for n, v in cv_scores.items()}).T
pd.DataFrame({f"{n} ROC-AUC": v["roc"] for n, v in cv_scores.items()} | {f"{n} PR-AUC": v["pr"] for n, v in cv_scores.items()} |
             {f"{n} recall": v["rec"] for n, v in cv_scores.items()},
             index=[f"fold {k}" for k in range(1, CV_FOLDS + 1)]).to_csv(out("other", "classification_cv_folds.csv"), index_label="fold")
cv.to_csv(out("other", "classification_cv_summary.csv"), index_label="Model")
print(cv.to_string())
fig, axes = plt.subplots(1, 3, figsize=(19, 5.0))
for ax, key, title, ylim in ((axes[0], "roc", "ROC-AUC", (0.4, 0.8)), (axes[1], "pr", "PR-AUC", (0.1, 0.4)),
                             (axes[2], "rec", f"recall at the {FLAG_RULE} threshold", (0.0, 1.0))):
    means = [np.mean(cv_scores[n][key]) for n in cv_scores]
    sds = [np.std(cv_scores[n][key]) for n in cv_scores]
    ax.bar(list(cv_scores), means, yerr=sds, color=[COLORS[n] for n in cv_scores], capsize=3)
    ax.set_ylim(*ylim); ax.set_title(f"{CV_FOLDS}-fold CV {title} (mean +/- std)", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=7.5)
fig.suptitle("Cross-validation on all 9,578 rows, every model (stratified folds; inner validation slice for early stopping)")
plots._save(fig, out("other", "classification_cv_roc_auc.png"))

# -----------------------------------------------------------------------------
# 12. Benchmark table (what the report needs): every model, every metric
# -----------------------------------------------------------------------------
section("12. QUANTITATIVE RESULTS & BENCHMARK  [artifact: benchmarking]")
table = bench.join(cv)
taught = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"]
taught_best_pr = table.loc[taught, "test_pr_auc"].max()
taught_best_roc = table.loc[taught, "test_roc_auc"].max()
table["PR-AUC gain vs best taught model"] = table["test_pr_auc"] - taught_best_pr
table["ROC-AUC gain vs best taught model"] = table["test_roc_auc"] - taught_best_roc
table["PR-AUC gain vs random"] = table["test_pr_auc"] - float(y_te.mean())   # PR-AUC of a random ranking = the base rate
table["recall gain vs plain 0.5"] = table["test_recall"] - table["test_recall@0.5"]
table.to_csv(out("other", "classification_benchmark_with_cv.csv"), index_label="Model")
rank_view = table[["test_roc_auc", "test_pr_auc", "CV ROC-AUC mean", "CV PR-AUC mean", f"CV recall@{FLAG_RULE} mean",
                   "PR-AUC gain vs random"]].sort_values("test_roc_auc", ascending=False).rename(columns={
    "test_roc_auc": "ROC-AUC", "test_pr_auc": "PR-AUC", "CV ROC-AUC mean": "CV ROC-AUC",
    "CV PR-AUC mean": "CV PR-AUC", f"CV recall@{FLAG_RULE} mean": "CV recall", "PR-AUC gain vs random": "PR-AUC over random"})
flag_view = table[["chosen_threshold", "test_precision", "test_recall", "test_f1", "test_fn",
                   "recall gain vs plain 0.5"]].sort_values("test_recall", ascending=False).rename(columns={
    "chosen_threshold": "threshold", "test_precision": "precision", "test_recall": "recall (caught)",
    "test_f1": "F1", "test_fn": "missed defaulters", "recall gain vs plain 0.5": "recall gain vs 0.5"})
print("\nA) RANKING quality (threshold-free) -- who orders the borrowers best?")
print(rank_view.round(4).to_string())
print(f"\nB) FLAGGING quality ({FLAG_RULE} threshold chosen on validation) -- who actually catches the defaulters?")
print(flag_view.round(4).to_string())
print(f"\nBest taught model: PR-AUC {taught_best_pr:.4f} / ROC-AUC {taught_best_roc:.4f}")
print(f"Better model (blend): PR-AUC {table.loc[BETTER, 'test_pr_auc']:.4f} / ROC-AUC {table.loc[BETTER, 'test_roc_auc']:.4f}")
rank_roc = table["test_roc_auc"].sort_values(ascending=False)
rank_rec = table["test_recall"].sort_values(ascending=False)
print(f"Ranking by test ROC-AUC (best first): {', '.join(rank_roc.index[:5])} ... (last: {rank_roc.index[-1]})")
print(f"Ranking by defaulters caught  (best first): {', '.join(rank_rec.index[:5])} ... (last: {rank_rec.index[-1]})")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 5.4), sharey=True)
for ax, col, title in ((a1, "test_roc_auc", "Test ROC-AUC"), (a2, "test_pr_auc", "Test PR-AUC")):
    order = table[col].sort_values()
    ax.barh(order.index, order.values, color=[COLORS[n] for n in order.index], height=0.62)
    ref = 0.5 if col == "test_roc_auc" else float(y_te.mean())
    ax.axvline(ref, color=plots.NEUTRAL, linestyle="--", label=f"random = {ref:.3f}")
    ax.set_title(title); ax.grid(axis="y", visible=False)
    for name, v in order.items():
        ax.text(v, name, f" {v:.3f}", va="center", fontsize=7)
    ax.set_xlim(0, order.max() * 1.14); ax.legend(fontsize=7)
fig.suptitle(f"Benchmark: every model on the same test set ({FLAG_RULE} thresholds chosen on validation)")
plots._save(fig, out("other", "classification_benchmark_all_metrics.png"))

# -----------------------------------------------------------------------------
# 13. Feature importance for every model (tree gain / permutation / linear coefficients / model agreement)
# -----------------------------------------------------------------------------
section("13. FEATURE IMPORTANCE, EVERY MODEL  [artifact: Hint - feature importance]")
imp_cols = {}
for name, m in fitted.items():
    try:
        tbl = dict(m.feature_importance_table())
        s = pd.Series(tbl).reindex(feature_cols)
        if s.abs().sum() > 0:
            imp_cols[name] = (s / s.abs().sum()).abs()          # normalise to a share, so models are comparable
    except Exception:
        pass
rng = np.random.RandomState(SEED)
base_auc = roc_auc(y_te, p_te["XGBoost"])
perm = []
for j in range(X_te.shape[1]):
    drops = []
    for _ in range(3):                                     # 3 shuffles per feature on the test set
        Xp = X_te.copy()
        Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
        drops.append(base_auc - roc_auc(y_te, fitted["XGBoost"].predict_proba(Xp)[:, 1]))
    perm.append(float(np.mean(drops)))
perm = pd.Series(perm, index=feature_cols).rename("xgboost_permutation_aucloss")
lin = pd.Series({r["feature"]: r["coef_per_1SD"] for r in fitted["Logistic Regression"].coefficient_table()}).rename("logistic_coef_per_1SD")
odds = pd.Series({r["feature"]: r["odds_ratio"] for r in fitted["Logistic Regression"].coefficient_table()}).rename("logistic_odds_ratio")
imp_all = pd.DataFrame(imp_cols).join(pd.DataFrame({"xgboost_permutation_aucloss": perm, "logistic_coef_per_1SD": lin, "logistic_odds_ratio": odds}))
imp_all = imp_all.sort_values("XGBoost", ascending=False)
imp_all.to_csv(out("other", "classification_feature_importance.csv"), index_label="feature")
print(f"  importance tables collected from: {', '.join(imp_cols)}")
print("\nTop 8 features by XGBoost gain share:")
print(imp_all["XGBoost"].head(8).to_string())
print("\nROC-AUC drop when the feature is shuffled (XGBoost, top 8):")
print(imp_all["xgboost_permutation_aucloss"].sort_values(ascending=False).head(8).to_string())
print("\nLogistic regression |coef| per 1 SD (top 8):")
print(imp_all["logistic_coef_per_1SD"].abs().sort_values(ascending=False).head(8).to_string())

gain = imp_cols.get("XGBoost", pd.Series(dtype=float))
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
for ax, s, title, fmt in ((axes[0], gain, "XGBoost gain share", "{:.1%}"),
                          (axes[1], perm, "ROC-AUC drop when shuffled (XGBoost)", "{:.3f}"),
                          (axes[2], lin.abs(), "Logistic |coef| per 1 SD", "{:.2f}")):
    s = s.sort_values(ascending=False).head(10)[::-1]
    col = COLORS["XGBoost"] if ax is not axes[2] else COLORS["Logistic Regression"]
    ax.barh(s.index, s.values, color=col, height=0.6)
    ax.set_title(title, fontsize=10); ax.grid(axis="y", visible=False)
    for name, v in list(s.items())[-3:]:
        ax.text(v, name, " " + fmt.format(v), va="center", fontsize=7)
    ax.set_xlim(0, s.max() * 1.25)
fig.suptitle("Feature importance by three different methods (they do not have to agree)")
plots._save(fig, out("other", "classification_feature_importance.png"))

# do the models agree on what matters? rank each feature inside every model and plot the ranks
hot = imp_all["XGBoost"].abs().sort_values(ascending=False).head(15).index
rank_mat = pd.DataFrame({n: imp_cols[n].reindex(hot).rank(ascending=False) for n in imp_cols})
fig, ax = plt.subplots(figsize=(0.55 * len(imp_cols) + 5, 6.2))
im = ax.imshow(rank_mat.values, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(imp_cols)), list(imp_cols), rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(len(hot)), hot, fontsize=8)
for (i, j), v in np.ndenumerate(rank_mat.values):
    ax.text(j, i, int(v), ha="center", va="center", fontsize=6.5)
fig.colorbar(im, ax=ax, shrink=0.7, label="rank inside the model (1 = most important)")
ax.set_title("Model agreement: where each of the 15 strongest features is ranked by every model")
plots._save(fig, out("other", "classification_feature_importance_agreement.png"))

# -----------------------------------------------------------------------------
# 14. Save + load the best model (highest validation PR-AUC), then score example borrowers
# -----------------------------------------------------------------------------
section("14. SAVE / LOAD THE BEST MODEL  [artifact: save-load model]")
best_name = max(fitted, key=lambda n: roc_auc(y_va, p_va[n]))     # the report targets ROC-AUC...
os.makedirs(os.path.join(ROOT, "TrainedModels"), exist_ok=True)
path = os.path.join(ROOT, "TrainedModels", "classification_model.joblib")
thr = float(bench.loc[best_name, "chosen_threshold"])
joblib.dump({"model": fitted[best_name], "name": best_name, "feature_cols": feature_cols, "target": "not.fully.paid",
             "threshold": thr, "threshold_rule": f"{FLAG_RULE} (chosen on validation)",
             "test_roc_auc": float(bench.loc[best_name, "test_roc_auc"]),
             "test_pr_auc": float(bench.loc[best_name, "test_pr_auc"]),
             "test_recall": float(bench.loc[best_name, "test_recall"])}, path)
bundle = joblib.load(path)
assert np.allclose(bundle["model"].predict_proba(X_te)[:, 1], p_te[best_name]), "reloaded model differs!"
print(f"Best model by validation ROC-AUC: {best_name} "
      f"(PR-AUC of the same model: {average_precision(y_va, p_va[best_name]):.4f})")
print(f"Saved + reloaded: {os.path.relpath(path, ROOT)}  (threshold {thr:.3f}, test ROC-AUC {bundle['test_roc_auc']:.4f}, "
      f"test PR-AUC {bundle['test_pr_auc']:.4f}, defaulters caught {bundle['test_recall']:.3f})")
risk = float(bundle["model"].predict_proba(X_te[:1])[0, 1])
print(f"Example borrower #1 of the test set: P(not fully paid) = {risk:.3f} -> "
      f"{'FLAG (high risk)' if risk >= thr else 'accept'} (threshold {thr:.3f}); true label = {int(y_te[0])}")

p_best = bundle["model"].predict_proba(X_te)[:, 1]
picked = list(np.argsort(-p_best)[:3]) + list(np.argsort(p_best)[:3])
ex = pd.DataFrame([{"test row": int(i), "P(not fully paid)": float(p_best[i]),
                    "decision at the threshold": "flag" if p_best[i] >= thr else "accept",
                    "true label": int(y_te[i])} for i in picked])
ex.to_csv(out("other", "classification_example_borrowers.csv"), index=False)
print(f"\nHighest / lowest predicted risk in the test set (threshold {thr:.3f}):")
print(ex.to_string(index=False))
print(f"At the chosen threshold the model flags {(p_best >= thr).sum()} of {len(p_best)} test borrowers "
      f"({(p_best >= thr).mean():.1%}), catching {(y_te[p_best >= thr]).sum()} of {int(y_te.sum())} defaulters "
      f"({(y_te[p_best >= thr]).sum() / y_te.sum():.1%}), while {y_te.mean():.1%} really defaulted")

# -----------------------------------------------------------------------------
# 15. Model-specific interpretability (perceptron variants, EBM curves, Naive Bayes densities, why-this-borrower)
# -----------------------------------------------------------------------------
section("15. MODEL-SPECIFIC INTERPRETABILITY  [artifact: analysis & discussion]")
pv_rows = []
for v in ("classic", "pocket", "averaged"):
    pm = PerceptronScratch(variant=v, learning_rate=0.1, n_epochs=40, random_state=SEED).fit(X_tr, y_tr, eval_set=[(X_va, y_va)], feature_names=feature_cols)
    pv_rows.append({"variant": v, "val ROC-AUC": roc_auc(y_va, pm.decision_function(X_va)), "test ROC-AUC": roc_auc(y_te, pm.decision_function(X_te)),
                    "mistakes in last epoch": pm.history_["mistakes"][-1], "converged": pm.converged_})
pv = pd.DataFrame(pv_rows).set_index("variant"); pv.to_csv(out("other", "classification_perceptron_variants.csv")); print("\n" + pv.to_string())
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.bar(pv.index, pv["mistakes in last epoch"], color=[COLORS["Perceptron (averaged)"] if v == "averaged" else plots.NEUTRAL for v in pv.index])
ax.set_title("Perceptron variants: mistakes in the last epoch (data is not linearly separable)")
plots._save(fig, out("other", "classification_perceptron_variants.png"))

# SVC (linear): the weights, and what the hinge had to do to catch defaulters
svc = fitted["SVC (linear)"]
svc_imp = pd.Series({f: w for f, w in svc.feature_importance_table()}).sort_values(ascending=False)
svc_imp.rename("weight_per_1SD").to_csv(out("other", "classification_svc_weights.csv"), index_label="feature")
fig, ax = plt.subplots(figsize=(7.5, 4.2))
sw = svc_imp.head(10)[::-1]
ax.barh(sw.index, sw.values, color=[plots.RED if v > 0 else plots.BLUE for v in sw.values])
ax.axvline(0, color=plots.NEUTRAL, linewidth=0.8)
ax.set_title("SVC (linear, balanced): weight per 1 SD (positive pushes towards default)")
plots._save(fig, out("other", "classification_svc_weights.png"))

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

# save the better models in their own files, reload them, explain one borrower
os.makedirs(os.path.join(ROOT, "TrainedModels"), exist_ok=True)
ebm_path = os.path.join(ROOT, "TrainedModels", "classification_ebm_model.joblib")
ebm.best_threshold_ = float(bench.loc["EBM (additive)", "chosen_threshold"]); ebm.save(ebm_path)
ebm2 = ExplainableBoostingScratch.load(ebm_path)
assert np.allclose(ebm2.predict_proba(X_te)[:, 1], p_te["EBM (additive)"]), "reloaded EBM differs!"
flagged = np.where((y_te == 1) & (ebm2.predict(X_te) == 1))[0]
i = int(flagged[0]) if len(flagged) else 0
exp = ebm2.explain(X_te[i:i + 1], top=6)
bl = fitted[BETTER]
bl_path = os.path.join(ROOT, "TrainedModels", "classification_blend_model.joblib")
bl.best_threshold_ = float(bench.loc[BETTER, "chosen_threshold"]); bl.save(bl_path)
bl2 = BlendEnsembleScratch.load(bl_path)
assert np.allclose(bl2.predict_proba(X_te)[:, 1], p_te[BETTER]), "reloaded blend differs!"
exb = bl2.explain(X_te[i:i + 1])
print(f"\nSaved + reloaded: {os.path.relpath(bl_path, ROOT)}  (reload gives identical predictions)")
print(f"Same borrower with the blend: P(EBM) = {exb['p_ebm']:.3f}, P(bagged MLP) = {exb['p_mlp_bag']:.3f} -> P(blend) = {exb['p_blend']:.3f}")
print(f"\nSaved + reloaded: {os.path.relpath(ebm_path, ROOT)}")
print(f"Borrower #{i} of the test set (true label {int(y_te[i])}): P(default) = {ebm2.predict_proba(X_te[i:i + 1])[0, 1]:.3f}  (threshold {ebm2.best_threshold_:.3f})")
print(f"  log-odds = intercept {exp['intercept']:+.3f} + contributions (sum = {exp['log_odds']:+.3f}):")
for f_, v_ in exp["contributions"]:
    print(f"     {f_:<32} {v_:+.3f}")

# -----------------------------------------------------------------------------
# 16. Batch runs: 10 fresh random 60/20/20 splits, EVERY model (seeds never used to design the blend)
# -----------------------------------------------------------------------------
section(f"16. BATCH RUNS: {N_BATCH} RANDOM SPLIT{'S' if N_BATCH > 1 else ''}, EVERY MODEL  "
        f"[artifact: training-testing iterations]")
BATCH_MODELS = dict(models)                       # all 15 models, nothing skipped
bres = {n: {"auc": [], "pr": [], "rec": [], "acc": []} for n in BATCH_MODELS}
for b in range(N_BATCH):
    s_tr, s_va, s_te, t_tr, t_va, t_te = train_val_test_split_scratch(X, y, seed=500 + b)
    t_split = time.time()
    failed_here = []
    for n, (factory, use_val) in BATCH_MODELS.items():
        try:
            t_m = time.time()
            m = fit_model(factory, use_val, s_tr, t_tr, s_va, t_va, feature_cols)
            p_te_b = m.predict_proba(s_te)[:, 1]
            thr_b, _ = choose_threshold(t_va, m.predict_proba(s_va)[:, 1], rule=FLAG_RULE, base_rate=float(t_va.mean()))
            bres[n]["auc"].append(roc_auc(t_te, p_te_b))
            bres[n]["pr"].append(average_precision(t_te, p_te_b))
            bres[n]["rec"].append(recall(t_te, p_te_b, thr_b))
            bres[n]["acc"].append(float(((p_te_b >= 0.5).astype(int) == t_te).mean()))
            if time.time() - t_m > 20:                     # only the slow models are worth reporting
                print(f"      {n} took {time.time() - t_m:.0f}s", flush=True)
        except Exception as e:                             # one bad model must never kill the whole batch
            failed_here.append(n)
            for k in ("auc", "pr", "rec", "acc"):
                bres[n][k].append(float("nan"))
            print(f"      {n} failed on this split: {type(e).__name__}: {str(e)[:80]}", flush=True)
    print(f"  split {b + 1}/{N_BATCH} done in {time.time() - t_split:.0f}s"
          + (f"  (failed: {', '.join(failed_here)})" if failed_here else ""), flush=True)
bt = pd.DataFrame({n: {"ROC-AUC mean": np.nanmean(v["auc"]), "ROC-AUC SD": np.nanstd(v["auc"]),
                       "PR-AUC mean": np.mean(v["pr"]), "PR-AUC SD": np.std(v["pr"]),
                       f"recall@{FLAG_RULE} mean": np.mean(v["rec"]), f"recall@{FLAG_RULE} SD": np.std(v["rec"]),
                       "accuracy@0.5 mean": np.mean(v["acc"])} for n, v in bres.items()}).T
bt.to_csv(out("other", "classification_batch_runs.csv"), index_label="Model"); print(bt.to_string())
rank_batch = bt["ROC-AUC mean"].sort_values(ascending=False)
print(f"\n  ranking over {N_BATCH} splits by ROC-AUC: {', '.join(rank_batch.index[:5])} ... (last: {rank_batch.index[-1]})")
for n in rank_batch.index[:6]:
    dlt = np.array(bres[n]["auc"]) - np.array(bres[rank_batch.index[0]]["auc"])
    ok = np.isfinite(dlt)
    print(f"  {n:<26} minus {rank_batch.index[0]:<26}: {np.nanmean(dlt):+.4f} AUC "
          f"(SD {np.nanstd(dlt):.4f}), better in {int((dlt[ok] > 0).sum())}/{int(ok.sum())} splits")
fig, axes = plt.subplots(1, 3, figsize=(19, 4.4))
for ax, k, t in ((axes[0], "auc", "ROC-AUC"), (axes[1], "pr", "PR-AUC"),
                 (axes[2], "rec", f"recall (defaulters caught) at the {FLAG_RULE} threshold")):
    bp = ax.boxplot([bres[n][k] for n in bres], tick_labels=list(bres), patch_artist=True, showmeans=True)
    for patch_, n in zip(bp["boxes"], bres):
        patch_.set_facecolor(COLORS[n]); patch_.set_alpha(0.7)
    ax.set_title(f"Test {t} over {N_BATCH} random splits", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=7.5)
plots._save(fig, out("other", "classification_batch_runs.png"))

# -----------------------------------------------------------------------------
# 17. Checklist of every result file (the report's quantitative-results checklist)
# -----------------------------------------------------------------------------
section("17. RESULT FILES WRITTEN  [report checklist]")
n_files = 0
for folder in sorted(os.listdir(RESULTS)):
    fp = os.path.join(RESULTS, folder)
    if not os.path.isdir(fp):
        continue
    files = sorted(os.listdir(fp))
    print(f"  Results/{folder}/  ({len(files)} files)")
    for f in files:
        print(f"      {f:<62} {os.path.getsize(os.path.join(fp, f)) / 1024:6.1f} KB")
    n_files += len(files)
tm = os.path.join(ROOT, "TrainedModels")
if os.path.isdir(tm):
    print("  TrainedModels/")
    for f in sorted(os.listdir(tm)):
        print(f"      {f:<62} {os.path.getsize(os.path.join(tm, f)) / 1024:6.1f} KB")
print(f"\n{n_files} files in Results/ + the models in TrainedModels/")
print(f"Models: {len(fitted)} scratch models | best by validation PR-AUC: {best_name} | "
      f"blend weight chosen on validation: {EBM_WEIGHT} | flagging rule: {FLAG_RULE}")
print("\n" + "=" * 78 + "\n  KEY NUMBERS (one-screen summary for the report)\n" + "=" * 78)
top20_lift = topk[(topk["Model"] == best_top20) & (topk["flagged share"] == 0.20)]["lift vs flagging at random"].iloc[0]
print(f"  data                : {len(df):,} borrowers, {len(feature_cols)} features, {y.mean():.1%} defaulters "
      f"(train {y_tr.mean():.1%} / val {y_va.mean():.1%} / test {y_te.mean():.1%})")
print(f"  models              : {len(fitted)} scratch models, every section covers all of them")
print(f"  accuracy warning    : answering 'everybody pays' scores {max(y_te.mean(), 1 - y_te.mean()):.1%} accuracy here, "
      f"so accuracy alone says nothing - use ROC-AUC / PR-AUC / recall")
print(f"  best ranking        : {best_roc:<28} ROC-AUC {bench.loc[best_roc, 'test_roc_auc']:.4f} | "
      f"PR-AUC {bench.loc[best_roc, 'test_pr_auc']:.4f} | CV ROC-AUC {table.loc[best_roc, 'CV ROC-AUC mean']:.4f}")
print(f"  most defaulters     : {best_recall:<28} catches {bench.loc[best_recall, 'test_recall']:.1%} "
      f"(misses {int(bench.loc[best_recall, 'test_fn'])} of {int(y_te.sum())})")
print(f"  better model (blend): ROC-AUC {bench.loc[BETTER, 'test_roc_auc']:.4f} | catches "
      f"{bench.loc[BETTER, 'test_recall']:.1%} | stack: EBM {EBM_WEIGHT:.2f} + bagged MLP {1 - EBM_WEIGHT:.2f} "
      f"(weight chosen by CV on the train set)")
print(f"  flagging rule       : '{FLAG_RULE}' - flags {rules['applicants flagged'].mean():.1%} of the applicants, "
      f"catches {rules[f'recall @{FLAG_RULE}'].mean():.1%} of the defaulters, flagged group "
      f"{rules['enrichment of the flagged group'].mean():.2f}x enriched")
print(f"  other rules (avg)   : 0.5 -> {ladder_tbl.loc['0.5 (plain)', 'recall (defaulters caught)']:.1%} recall "
      f"/ {ladder_tbl.loc['0.5 (plain)', 'applicants flagged']:.1%} flagged | f1 -> "
      f"{ladder_tbl.loc['f1', 'recall (defaulters caught)']:.1%} / {ladder_tbl.loc['f1', 'applicants flagged']:.1%} | "
      f"f2 (catch-max) -> {ladder_tbl.loc['f2', 'recall (defaulters caught)']:.1%} / "
      f"{ladder_tbl.loc['f2', 'applicants flagged']:.1%} flagged (only "
      f"{ladder_tbl.loc['f2', 'enrichment vs random']:.2f}x enriched)")
print(f"  shortlist (no cut-off needed): flagging the riskiest 20% catches {pivot.loc[0.20].max():.1%} of all "
      f"defaulters with {best_top20} (lift {top20_lift:.2f}x); the riskiest 30% catches {pivot.loc[0.30].max():.1%}")
print(f"  verification        : every scratch model checked against scikit-learn (section 10); probabilities have no "
      f"NaN and stay in [0,1]; every flagging threshold comes from the validation set")
print("\nDONE.")
