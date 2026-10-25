"""
The standard RESULTS section of every regression notebook: metrics, charts (shown in the notebook
and saved to Results/Regression/<notebook>/), and the files the comparison notebook reads.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plots
from metrics import mae, mse, r2, rmse


def regression_row(name, y_train, p_train, y_test, p_test):
    return {
        "Model": name,
        "Train loss (MSE)": mse(y_train, p_train),
        "Test loss (MSE)": mse(y_test, p_test),
        "Test RMSE": rmse(y_test, p_test),
        "Test MAE": mae(y_test, p_test),
        "Test R2": r2(y_test, p_test),
    }


def plot_loss_curve(name, history, path):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(np.arange(1, len(history) + 1), history, color=plots.MODEL_COLORS[name])
    ax.set_yscale("log")
    ax.set_xlabel("iteration (gradient-descent step / boosting tree)")
    ax.set_ylabel("training MSE (log scale)")
    ax.set_title(f"{plots.SHORT[name]}: training loss curve (final MSE {history[-1]:.6f})")
    plots._save(fig, path)


def plot_actual_vs_predicted(name, y_test, p_test, path):
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.scatter(y_test * 100, p_test * 100, s=8, alpha=0.35, color=plots.MODEL_COLORS[name], linewidths=0)
    lo, hi = float(min(y_test.min(), p_test.min())) * 100, float(max(y_test.max(), p_test.max())) * 100
    ax.plot([lo, hi], [lo, hi], color=plots.INK_2, linewidth=1.2, label="perfect prediction")
    ax.set_xlabel("actual int.rate (%)")
    ax.set_ylabel("predicted int.rate (%)")
    ax.set_title(f"{plots.SHORT[name]}: actual vs predicted (test R2 {r2(y_test, p_test):.3f})")
    ax.legend(loc="upper left")
    plots._save(fig, path)


def plot_residuals(name, y_test, p_test, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(p_test * 100, (y_test - p_test) * 100, s=8, alpha=0.3, color=plots.MODEL_COLORS[name], linewidths=0)
    ax.axhline(0, color=plots.INK_2, linewidth=1.2)
    ax.set_xlabel("predicted int.rate (%)")
    ax.set_ylabel("residual = actual - predicted (pts)")
    ax.set_title(f"{plots.SHORT[name]}: residuals (should scatter randomly around 0)")
    plots._save(fig, path)


def report_regressor(out, name, model, d, history, check):
    """Print + chart + save everything for one fitted regression pipeline. Returns its metrics row.

    check = (scikit-learn test R2, short description of the library model)
    """
    p_train, p_test = model.predict(d.X_train), model.predict(d.X_test)
    row = regression_row(name, d.y_train, p_train, d.y_test, p_test)
    row["scikit-learn R2"], row["Difference"] = float(check[0]), float(row["Test R2"] - check[0])
    print(pd.Series(row).drop("Model").to_string(float_format=lambda v: f"{v:.6f}"))
    print(f"\nscratch vs {check[1]}: R2 {row['Test R2']:.4f} vs {check[0]:.4f} (difference {row['Difference']:+.4f})")

    plot_loss_curve(name, history, out.image("loss_curve.png"))
    plot_actual_vs_predicted(name, d.y_test, p_test, out.image("actual_vs_predicted.png"))
    plot_residuals(name, d.y_test, p_test, out.image("residuals.png"))

    pd.DataFrame([row]).to_csv(out.table("metrics.csv"), index=False)
    pd.DataFrame({"iteration": np.arange(1, len(history) + 1), "train_loss_mse": history}).to_csv(
        out.table("loss_curve.csv"), index=False
    )
    np.savez(out.table("predictions.npz"), p_train=p_train, p_test=p_test)
    joblib.dump(
        {
            "model": model,
            "name": name,
            "target": "int.rate",
            "feature_cols": d.feature_cols,
            "test_r2": row["Test R2"],
            "test_rmse": row["Test RMSE"],
            "test_mae": row["Test MAE"],
        },
        out.model(),
    )
    back = joblib.load(out.model())["model"]
    same = bool(np.allclose(back.predict(d.X_test), p_test))
    assert same, "the reloaded model gives different predictions!"
    print(f"saved + reloaded Results/Regression/{out.name}/model.joblib (identical predictions: {same})")
    return row
