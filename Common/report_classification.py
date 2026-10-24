"""
The standard RESULTS section of every classification notebook: one call prints the metrics,
draws the charts (shown in the notebook and saved to Images/Classification/<notebook>/) and
saves what the comparison notebook needs (predictions, metrics, scratch-vs-sklearn check, model).
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plots
from evaluation_classification import COLORS, FLAG_RULE, evaluate_model, staged_pos
from metrics_classification import (
    average_precision,
    confusion_matrix,
    logloss,
    precision_recall_curve_scratch,
    roc_curve_scratch,
)


def sklearn_check(label, difference, auc_scratch, auc_sklearn, meaning):
    """One row of the scratch-vs-library verification table."""
    return {
        "check": label,
        "difference": float(difference),
        "AUC scratch": float(auc_scratch),
        "AUC sklearn": float(auc_sklearn),
        "the difference is": meaning,
    }


def predict_all(model, d):
    return {
        "p_tr": model.predict_proba(d.X_tr)[:, 1],
        "p_va": model.predict_proba(d.X_va)[:, 1],
        "p_te": model.predict_proba(d.X_te)[:, 1],
    }


def metrics_table(row):
    """The headline numbers of one model as a small readable table."""
    keys = [
        ("threshold (chosen on validation)", "chosen_threshold"),
        ("log loss", "test_logloss"),
        ("accuracy", "test_accuracy"),
        ("precision", "test_precision"),
        ("recall / sensitivity", "test_recall"),
        ("specificity", "test_specificity"),
        ("F1", "test_f1"),
        ("ROC-AUC", "test_roc_auc"),
        ("PR-AUC", "test_pr_auc"),
        ("defaulters missed (FN)", "test_fn"),
    ]
    t = pd.DataFrame({"test set": [row[k] for _, k in keys]}, index=[n for n, _ in keys])
    t["at plain 0.5"] = [
        row.get(k + "@0.5", np.nan) if k.startswith("test_") else (0.5 if k == "chosen_threshold" else np.nan)
        for _, k in keys
    ]
    return t


def plot_loss_curve(name, model, row, path):
    """Training log loss per iteration; models fitted in one pass (trees, Naive Bayes) show one point."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    h = list(getattr(model, "train_loss", None) or [])
    if h:
        ax.plot(np.arange(1, len(h) + 1), h, color=COLORS[name])
        ax.plot([len(h)], [h[-1]], "o", color=COLORS[name])
        ax.set_xlabel("iteration")
        ax.set_title(f"{name}: training log loss ({len(h)} iterations, final {h[-1]:.4f})")
    else:
        v = row["train_logloss"]
        ax.axhline(v, color=COLORS[name], linewidth=2)
        ax.set_xticks([])
        ax.set_xlabel("single fit (no per-iteration history)")
        ax.set_title(f"{name}: training log loss = {v:.4f} (fitted in one pass)")
    ax.set_ylabel("train log loss (lower is better)")
    plots._save(fig, path)


def plot_performance_curve(name, model, d, path):
    """Validation log loss and PR-AUC after every training stage (only models with staged predictions)."""
    st = staged_pos(model, d.X_va)
    ll = [logloss(d.y_va, p) for p in st]
    pr = [average_precision(d.y_va, p) for p in st]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(np.arange(1, len(ll) + 1), ll, color=plots.BLUE, label="validation log loss")
    ax.set_xlabel("training stage (tree / gradient step)")
    ax.set_ylabel("validation log loss", color=plots.BLUE)
    ax.axvline(
        int(np.argmin(ll)) + 1,
        color=plots.NEUTRAL,
        linestyle=":",
        label=f"best log loss at stage {int(np.argmin(ll)) + 1}",
    )
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, len(pr) + 1), pr, color=plots.RED, label="validation PR-AUC")
    ax2.set_ylabel("validation PR-AUC", color=plots.RED)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    ax.set_title(f"{name}: performance curve on the validation set")
    plots._save(fig, path)
    return pd.DataFrame({"stage": np.arange(1, len(ll) + 1), "val_logloss": ll, "val_pr_auc": pr})


def plot_confusion(name, d, p_te, thr, path):
    cm = confusion_matrix(d.y_te, p_te, thr)
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=12, color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["predicted paid (0)", "predicted default (1)"], fontsize=8)
    ax.set_yticks([0, 1], ["paid (0)", "default (1)"], fontsize=8)
    ax.set_title(f"{name}: confusion matrix (test)\nthreshold {thr:.3f} ({FLAG_RULE} on validation)", fontsize=9)
    plots._save(fig, path)
    return cm


def plot_roc_pr(name, d, p_te, row, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    fpr, tpr, _ = roc_curve_scratch(d.y_te, p_te)
    a1.plot(fpr, tpr, color=COLORS[name], label=f"{name} (AUC {row['test_roc_auc']:.4f})")
    a1.plot([0, 1], [0, 1], color=plots.NEUTRAL, linestyle="--", label="random (0.500)")
    a1.set_xlabel("false positive rate")
    a1.set_ylabel("true positive rate (recall)")
    a1.set_title("ROC curve (test)")
    a1.legend(fontsize=8)
    prec, rec, _ = precision_recall_curve_scratch(d.y_te, p_te)
    a2.plot(rec, prec, color=COLORS[name], label=f"{name} (PR-AUC {row['test_pr_auc']:.4f})")
    a2.axhline(d.y_te.mean(), color=plots.NEUTRAL, linestyle="--", label=f"random = base rate {d.y_te.mean():.3f}")
    a2.set_xlabel("recall (defaulters caught)")
    a2.set_ylabel("precision")
    a2.set_title("Precision-recall curve (test)")
    a2.legend(fontsize=8)
    plots._save(fig, path)


def plot_importance(name, imp, path, top=10):
    s = imp.abs().sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(s.index, s.values, color=COLORS[name], height=0.6)
    for f, v in s.items():
        ax.text(v, f, f" {v:.1%}", va="center", fontsize=7)
    ax.set_xlim(0, s.max() * 1.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of the total importance")
    ax.set_title(f"{name}: top {top} features")
    plots._save(fig, path)


def save_model(out, name, model, row, feature_cols):
    """joblib file in TrainedModels/Classification/, reloaded once to prove it gives identical predictions."""
    model.best_threshold_ = row["chosen_threshold"]  # so .predict() after a reload uses the same threshold
    meta = {
        "name": name,
        "feature_cols": feature_cols,
        "target": "not.fully.paid",
        "threshold": row["chosen_threshold"],
        "threshold_rule": f"{FLAG_RULE} (chosen on validation)",
        "test_roc_auc": row["test_roc_auc"],
        "test_pr_auc": row["test_pr_auc"],
        "test_logloss": row["test_logloss"],
        "test_recall": row["test_recall"],
    }
    if hasattr(model, "get_state") and hasattr(type(model), "from_state"):
        # models holding factories/lambdas (Stacking) cannot be pickled as objects: store their state instead
        joblib.dump({"model": type(model).__name__, "state": model.get_state(), **meta}, out.model())
        return type(model).from_state(joblib.load(out.model())["state"])
    joblib.dump({"model": model, **meta}, out.model())
    return joblib.load(out.model())["model"]


def report_classifier(out, name, model, d, check=None):
    """Print + chart + save everything for one fitted classification model. Returns its metrics row."""
    p = predict_all(model, d)
    row = {
        "Model": name,
        **evaluate_model(d.y_va, p["p_va"], d.y_te, p["p_te"], rule=FLAG_RULE, base_rate=float(d.y_va.mean())),
    }
    row["train_logloss"] = logloss(d.y_tr, p["p_tr"])
    row["val_logloss"] = logloss(d.y_va, p["p_va"])
    print(metrics_table(row).to_string(float_format=lambda v: f"{v:.4f}"))
    print(
        f"\ncatches {row['test_recall']:.1%} of the {int(d.y_te.sum())} test defaulters "
        f"({int(row['test_fn'])} missed), flags {(p['p_te'] >= row['chosen_threshold']).mean():.1%} of applicants"
    )

    plot_loss_curve(name, model, row, out.image("loss_curve.png"))
    if hasattr(model, "staged_predict_proba"):
        plot_performance_curve(name, model, d, out.image("performance_curve.png")).to_csv(
            out.table("performance_curve.csv"), index=False
        )
    cm = plot_confusion(name, d, p["p_te"], row["chosen_threshold"], out.image("confusion_matrix.png"))
    plot_roc_pr(name, d, p["p_te"], row, out.image("roc_pr_curves.png"))
    imp = None
    try:
        tbl = pd.Series(dict(model.feature_importance_table())).reindex(d.feature_cols)
        if tbl.abs().sum() > 0:
            imp = (tbl / tbl.abs().sum()).abs()  # normalised share, so models are comparable
    except (AttributeError, TypeError, NotImplementedError):
        pass
    if imp is not None:
        plot_importance(name, imp, out.image("feature_importance.png"))
        imp.rename("importance_share").to_csv(out.table("feature_importance.csv"), index_label="feature")

    if check is not None:
        print("\nscratch vs library:")
        for c in check if isinstance(check, list) else [check]:
            print(
                f"  {c['check']}\n    AUC scratch {c['AUC scratch']:.4f} | AUC library {c['AUC sklearn']:.4f} | "
                f"{c['the difference is']} = {c['difference']:.3g}"
            )
        out.save_json("check.json", check)

    pd.DataFrame([row]).to_csv(out.table("metrics.csv"), index=False)
    pd.DataFrame(cm, index=["true 0", "true 1"], columns=["pred 0", "pred 1"]).to_csv(out.table("confusion_matrix.csv"))
    if getattr(model, "train_loss", None):
        pd.DataFrame(
            {"iteration": np.arange(1, len(model.train_loss) + 1), "train_logloss": list(model.train_loss)}
        ).to_csv(out.table("loss_curve.csv"), index=False)
    np.savez(out.table("predictions.npz"), **p)
    back = save_model(out, name, model, row, d.feature_cols)
    same = bool(np.allclose(back.predict_proba(d.X_te)[:, 1], p["p_te"]))
    assert same, "the reloaded model gives different predictions!"
    print(f"\nsaved + reloaded TrainedModels/Classification/{out.name}.joblib (identical predictions: {same})")
    return row
