"""
Run the whole REGRESSION task (target: int.rate):
  load data -> data analysis -> features -> split/scale -> train the 4 scratch models (+ sklearn versions)
  -> loss / R2 / RMSE / MAE -> 5-fold cross-validation -> feature importance -> save + load the final model.

Run:  python run_regression.py        (from the project root)
Outputs: printed tables, Results/*.csv (one file per table), TrainedModels/regression_model.joblib
"""
import itertools
import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "Common"))
sys.path.insert(0, os.path.join(ROOT, "RegressionAlgorithms"))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures

from BetterModel_GradientBoostingRegressor import GradientBoostingRegressorScratch
from data import (StandardScalerScratch, build_regression_data, kfold_indices, load_dataset,
                  train_test_split_scratch)
from LinearRegression import LinearRegressionScratch
from metrics import mae, mse, permutation_importance, r2, rmse
from MultipleRegression import MultipleRegressionScratch
from PolynomialRegression import PolynomialRegressionScratch

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:.6f}")
SEED = 42
RESULTS = os.path.join(ROOT, "Results")   # every table below is also saved here as a CSV
os.makedirs(RESULTS, exist_ok=True)
POLY_FEATURES = ["fico", "revol.util", "dti", "inq.last.6mths", "credit.policy"]


def section(title):
    print("\n" + "=" * 70 + f"\n  {title}\n" + "=" * 70)


# -----------------------------------------------------------------------------
# 1. Data analysis
# -----------------------------------------------------------------------------
section("1. DATA ANALYSIS  [artifact: Task / Data analysis, feature insight]")
df = load_dataset()
print(f"Rows x columns : {df.shape}   |   missing values: {int(df.isnull().sum().sum())}")
print(f"int.rate       : mean={df['int.rate'].mean():.4f}  std={df['int.rate'].std():.4f}  "
      f"min={df['int.rate'].min():.4f}  max={df['int.rate'].max():.4f}")
X, y, feature_cols = build_regression_data(df)
print(f"Features used  : {len(feature_cols)} (target int.rate; not.fully.paid excluded = future outcome / leakage)")
corr = pd.Series({c: np.corrcoef(X[:, i], y)[0, 1] for i, c in enumerate(feature_cols)})
pd.DataFrame({"statistic": ["rows", "columns", "missing_values", "int.rate_mean", "int.rate_std", "int.rate_min", "int.rate_max"],
              "value": [len(df), df.shape[1], int(df.isnull().sum().sum()), df["int.rate"].mean(), df["int.rate"].std(),
                        df["int.rate"].min(), df["int.rate"].max()]}).to_csv(os.path.join(RESULTS, "01_data_analysis_summary.csv"), index=False)
corr.reindex(corr.abs().sort_values(ascending=False).index).rename("correlation_with_int.rate").to_csv(
    os.path.join(RESULTS, "01_data_analysis_correlation.csv"), index_label="feature")
print("\nTop correlations with int.rate:")
print(corr.reindex(corr.abs().sort_values(ascending=False).index).head(8).to_string())

# -----------------------------------------------------------------------------
# 2. Split + scale (scaler learns from TRAIN only)
# -----------------------------------------------------------------------------
section("2. TRAIN / TEST SPLIT (80/20) AND SCALING  [artifact: training-testing iterations]")
X_train, X_test, y_train, y_test = train_test_split_scratch(X, y, test_size=0.2, seed=SEED)
scaler = StandardScalerScratch().fit(X_train)
Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)
fico = feature_cols.index("fico")
poly_idx = [feature_cols.index(f) for f in POLY_FEATURES]
print(f"Train {X_train.shape}   Test {X_test.shape}")

# -----------------------------------------------------------------------------
# 3. Hyper-parameters of the better model: chosen by 3-fold CV on the TRAIN set only
#    (sklearn GBR is used as a fast ready-to-use stand-in for the search)
# -----------------------------------------------------------------------------
section("3. HYPER-PARAMETER SEARCH FOR THE BETTER MODEL  [CV on the train set only]")
best = None
for n_est, depth, lr in itertools.product([300, 600], [3, 4], [0.05, 0.1]):
    sk = GradientBoostingRegressor(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                                   subsample=0.8, min_samples_leaf=5, random_state=SEED)
    score = cross_val_score(sk, X_train, y_train, cv=KFold(3, shuffle=True, random_state=SEED), scoring="r2").mean()
    print(f"  n_estimators={n_est:<4} max_depth={depth} learning_rate={lr:<5} -> CV R2 = {score:.4f}")
    if best is None or score > best[0]:
        best = (score, n_est, depth, lr)
_, GBR_N, GBR_DEPTH, GBR_LR = best
print(f"Chosen: n_estimators={GBR_N}, max_depth={GBR_DEPTH}, learning_rate={GBR_LR}, subsample=0.8")


# -----------------------------------------------------------------------------
# 4. The four scratch models, each as (fit_and_predict function on given train/test arrays)
# -----------------------------------------------------------------------------
def make_models():
    """name -> function(X_tr, y_tr, X_te) returning (fitted_model, train_pred, test_pred, transform_fn)."""

    def simple(Xtr, ytr, Xte):
        sc = StandardScalerScratch().fit(Xtr[:, [fico]])
        m = LinearRegressionScratch(0.1, 2000).fit_gradient_descent(sc.transform(Xtr[:, [fico]]), ytr)
        f = lambda A: m.predict(sc.transform(A[:, [fico]]))
        return m, f(Xtr), f(Xte), f

    def multiple(Xtr, ytr, Xte):
        sc = StandardScalerScratch().fit(Xtr)
        m = MultipleRegressionScratch(0.1, 2000).fit_gradient_descent(sc.transform(Xtr), ytr)
        f = lambda A: m.predict(sc.transform(A))
        return m, f(Xtr), f(Xte), f

    def poly(Xtr, ytr, Xte):
        sc = StandardScalerScratch().fit(Xtr[:, poly_idx])
        m = PolynomialRegressionScratch(degree=2, learning_rate=0.05, n_iterations=3000)
        m.fit(sc.transform(Xtr[:, poly_idx]), ytr)
        f = lambda A: m.predict(sc.transform(A[:, poly_idx]))
        return m, f(Xtr), f(Xte), f

    def better(Xtr, ytr, Xte):
        m = GradientBoostingRegressorScratch(GBR_N, GBR_LR, GBR_DEPTH, subsample=0.8, random_state=SEED).fit(Xtr, ytr)
        return m, m.predict(Xtr), m.predict(Xte), m.predict

    return {"1. Simple Linear (FICO)": simple, "2. Multiple Linear": multiple,
            "3. Polynomial (deg 2)": poly, "4. Gradient Boosting (better)": better}


section("4. TRAINING THE 4 SCRATCH MODELS + BASELINE  [artifact: A. Regression]")
models = make_models()
fitted, rows = {}, []
baseline = np.full_like(y_test, y_train.mean())
rows.append({"Model": "Baseline (predict the mean)", "Train loss (MSE)": mse(y_train, np.full_like(y_train, y_train.mean())),
             "Test loss (MSE)": mse(y_test, baseline), "Test RMSE": rmse(y_test, baseline),
             "Test MAE": mae(y_test, baseline), "Test R2": r2(y_test, baseline)})
for name, fn in models.items():
    t0 = time.time()
    model, p_tr, p_te, predict_fn = fn(X_train, y_train, X_test)
    fitted[name] = (model, predict_fn)
    rows.append({"Model": name, "Train loss (MSE)": mse(y_train, p_tr), "Test loss (MSE)": mse(y_test, p_te),
                 "Test RMSE": rmse(y_test, p_te), "Test MAE": mae(y_test, p_te), "Test R2": r2(y_test, p_te)})
    print(f"  trained {name:<32} in {time.time() - t0:5.1f}s")
results = pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 5. Verification against ready-to-use (scikit-learn) versions
# -----------------------------------------------------------------------------
section("5. VERIFICATION: scratch vs scikit-learn  [artifact: ready-to-use function check]")
sk_multi = SkLinearRegression().fit(X_train, y_train)
sk_simple = SkLinearRegression().fit(X_train[:, [fico]], y_train)
pf = PolynomialFeatures(2, include_bias=False)
sc_p = StandardScalerScratch().fit(X_train[:, poly_idx])
sk_poly_m = SkLinearRegression().fit(pf.fit_transform(sc_p.transform(X_train[:, poly_idx])), y_train)
sk_poly_pred = sk_poly_m.predict(pf.transform(sc_p.transform(X_test[:, poly_idx])))
sk_gbr = GradientBoostingRegressor(n_estimators=GBR_N, max_depth=GBR_DEPTH, learning_rate=GBR_LR, subsample=0.8,
                                   min_samples_leaf=5, random_state=SEED).fit(X_train, y_train)
checks = {
    "1. Simple Linear (FICO)": r2(y_test, sk_simple.predict(X_test[:, [fico]])),
    "2. Multiple Linear": r2(y_test, sk_multi.predict(X_test)),
    "3. Polynomial (deg 2)": r2(y_test, sk_poly_pred),
    "4. Gradient Boosting (better)": r2(y_test, sk_gbr.predict(X_test)),
}
ver = pd.DataFrame({"Scratch R2": results.set_index("Model").loc[list(checks), "Test R2"],
                    "scikit-learn R2": pd.Series(checks)})
ver["Difference"] = ver["Scratch R2"] - ver["scikit-learn R2"]
ver.to_csv(os.path.join(RESULTS, "03_verification_scratch_vs_sklearn.csv"), index_label="Model")
print(ver.to_string())
print("(small differences are expected: gradient descent vs closed form; boosting uses different random subsamples)")

# -----------------------------------------------------------------------------
# 6. Repeated train/test runs: 5-fold cross-validation on all data
# -----------------------------------------------------------------------------
section("6. 5-FOLD CROSS-VALIDATION  [artifact: batch runs]  (mean +/- std of test R2)")
cv_scores = {name: [] for name in models}
for k, (tr, te) in enumerate(kfold_indices(len(y), 5, SEED), 1):
    for name, fn in models.items():
        _, _, p_te, _ = fn(X[tr], y[tr], X[te])
        cv_scores[name].append(r2(y[te], p_te))
    print(f"  fold {k}/5 done")
cv = pd.DataFrame({"CV R2 mean": {n: np.mean(v) for n, v in cv_scores.items()},
                   "CV R2 std": {n: np.std(v) for n, v in cv_scores.items()}})
pd.DataFrame(cv_scores, index=[f"fold {k}" for k in range(1, 6)]).to_csv(os.path.join(RESULTS, "04_cross_validation_folds.csv"), index_label="fold")
print(cv.to_string())

# -----------------------------------------------------------------------------
# 7. Benchmark table (what the report needs): every model, every metric
# -----------------------------------------------------------------------------
section("7. QUANTITATIVE RESULTS & BENCHMARK  [artifact: Loss, R-Square, benchmarking]")
table = results.set_index("Model").join(cv)
taught_best = table.loc[["1. Simple Linear (FICO)", "2. Multiple Linear", "3. Polynomial (deg 2)"], "Test R2"].max()
table["R2 gain vs best taught model"] = table["Test R2"] - taught_best
print(table.to_string())
print(f"\nBest taught model test R2 = {taught_best:.4f}; better model = "
      f"{table.loc['4. Gradient Boosting (better)', 'Test R2']:.4f}")
table.to_csv(os.path.join(RESULTS, "02_benchmark_loss_r2.csv"))

# -----------------------------------------------------------------------------
# 8. Loss curves (numbers only; charts come later)
# -----------------------------------------------------------------------------
section("8. LOSS CURVE + PERFORMANCE CURVE  [artifact: Loss, performance curve] (numbers; charts later)")
gd_models = {"1. Simple Linear (FICO)": fitted["1. Simple Linear (FICO)"][0].loss_history,
             "2. Multiple Linear": fitted["2. Multiple Linear"][0].loss_history,
             "3. Polynomial (deg 2)": fitted["3. Polynomial (deg 2)"][0].loss_history,
             "4. Gradient Boosting (better)": fitted["4. Gradient Boosting (better)"][0].train_loss}
for name, hist in gd_models.items():
    marks = [0, len(hist) // 4, len(hist) // 2, len(hist) - 1]
    print(f"  {name:<32} " + "  ".join(f"it {m + 1}: {hist[m]:.6f}" for m in marks))
gbr_model = fitted["4. Gradient Boosting (better)"][0]
pd.DataFrame([{"model": n, "iteration": i + 1, "train_loss_mse": v} for n, h in gd_models.items() for i, v in enumerate(h)]).to_csv(
    os.path.join(RESULTS, "05_loss_curves.csv"), index=False)
staged = [r2(y_test, p) for p in gbr_model.staged_predict(X_test)]
pd.DataFrame({"n_trees": range(len(staged)), "test_r2": staged}).to_csv(os.path.join(RESULTS, "06_performance_curve_boosting.csv"), index=False)
print(f"  Boosting test-R2 performance curve: peak {max(staged):.4f} at tree {int(np.argmax(staged))} of {GBR_N}")

# -----------------------------------------------------------------------------
# 9. Feature importance
# -----------------------------------------------------------------------------
section("9. FEATURE IMPORTANCE  [artifact: Hint - feature importance]")
imp_gbr = pd.Series(gbr_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("Gradient Boosting (share of total SSE reduction):")
print(imp_gbr.head(8).to_string())
multi_model = fitted["2. Multiple Linear"][0]
imp_lin = pd.Series(multi_model.standardized_importance(Xs_train) , index=feature_cols).sort_values(ascending=False)
print("\nMultiple Linear (|coef| per 1 SD of the feature):")
print(imp_lin.head(8).to_string())
perm = pd.Series(permutation_importance(gbr_model.predict, X_test, y_test), index=feature_cols).sort_values(ascending=False)
print("\nPermutation importance of Gradient Boosting (rise in test MSE when the feature is shuffled):")
pd.DataFrame({"gradient_boosting_gain_share": imp_gbr, "multiple_linear_std_coef": imp_lin, "gradient_boosting_permutation": perm}).sort_values(
    "gradient_boosting_gain_share", ascending=False).to_csv(os.path.join(RESULTS, "07_feature_importance.csv"), index_label="feature")
print(perm.head(8).to_string())

# -----------------------------------------------------------------------------
# 10. Save and load the final model, then predict a new applicant
# -----------------------------------------------------------------------------
section("10. SAVE / LOAD THE FINAL MODEL  [artifact: save-load model]")
os.makedirs(os.path.join(ROOT, "TrainedModels"), exist_ok=True)
path = os.path.join(ROOT, "TrainedModels", "regression_model.joblib")
joblib.dump({"model": gbr_model, "feature_cols": feature_cols, "target": "int.rate",
             "test_r2": float(r2(y_test, gbr_model.predict(X_test)))}, path)
bundle = joblib.load(path)
print(f"Saved + reloaded: {os.path.relpath(path, ROOT)}  (test R2 stored = {bundle['test_r2']:.4f})")
assert np.allclose(bundle["model"].predict(X_test), gbr_model.predict(X_test)), "reloaded model differs!"
applicant = {"credit.policy": 1, "installment": 300.0, "log.annual.inc": np.log(65000), "dti": 12.5, "fico": 720,
             "revol.bal": 8000, "revol.util": 40.0, "inq.last.6mths": 1, "delinq.2yrs": 0, "pub.rec": 0,
             "installment_to_income": 300.0 / 65000, "high_revol_util": 0, "bad_history_flag": 0,
             "years_with_cr_line": 10.0}
x_new = np.array([[applicant.get(c, 0) for c in bundle["feature_cols"]]], dtype=float)
rate = float(bundle["model"].predict(x_new)[0])
print(f"Example applicant (FICO 720, DTI 12.5, income $65k): predicted int.rate = {rate:.4f} ({rate * 100:.2f}%)")
print("\nDONE.")
