"""
Shared chart code (matplotlib). Every function draws ONE chart and saves it as a PNG.

Style rules (from the dataviz guidelines): a model keeps the same colour in every chart, one series = one colour,
hairline solid grid, thin marks, no top/right frame, values labelled only where it matters (tips / extremes).
"""
import matplotlib

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, RED, MID_GRAY, NEUTRAL = "#2a78d6", "#e34948", "#f0efec", "#8a8985"

# One fixed colour per model (categorical slots 1-4); the colour follows the model, never its rank.
MODEL_COLORS = {
    "1. Simple Linear (FICO)": "#2a78d6",
    "2. Multiple Linear": "#eb6834",
    "3. Polynomial (deg 2)": "#1baf7a",
    "4. Gradient Boosting (better)": "#eda100",
}
SHORT = {
    "1. Simple Linear (FICO)": "Simple Linear",
    "2. Multiple Linear": "Multiple Linear",
    "3. Polynomial (deg 2)": "Polynomial",
    "4. Gradient Boosting (better)": "Gradient Boosting",
}
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", [BLUE, MID_GRAY, RED])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 1.8,
})


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _tint(color, amount=0.55):
    """Lighter version of a colour (mix with the surface) - used for 'train' next to a solid 'test'."""
    r, g, b = to_rgb(color)
    s = to_rgb(SURFACE)
    return tuple(c + (sc - c) * amount for c, sc in zip((r, g, b), s))


# ----------------------------------------------------------------------------------------------
# 01  DATA ANALYSIS
# ----------------------------------------------------------------------------------------------
def plot_target_distribution(y, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y * 100, bins=40, color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.axvline(y.mean() * 100, color=INK_2, linewidth=1.2)
    ax.text(y.mean() * 100 + 0.2, ax.get_ylim()[1] * 0.93, f"mean {y.mean() * 100:.2f}%", color=INK_2, va="top")
    ax.set_xlabel("Interest rate (%)")
    ax.set_ylabel("Number of loans")
    ax.set_title("Most loans sit around a 12% interest rate")
    _save(fig, path)


def plot_correlation_bars(corr, path):
    """corr: Series (feature -> correlation with int.rate). Diverging: blue = lowers the rate, red = raises it."""
    c = corr.sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(c.index, c.values, color=[BLUE if v < 0 else RED for v in c.values], height=0.6)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    for name in list(c.abs().sort_values(ascending=False).index[:3]):
        v = c[name]
        ax.text(v + (0.015 if v >= 0 else -0.015), name, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", color=INK)
    ax.set_xlim(-0.85, 0.65)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Correlation with int.rate   (blue = lower rate, red = higher rate)")
    ax.set_title("FICO is by far the strongest signal for the interest rate")
    _save(fig, path)


def plot_correlation_heatmap(X, feature_cols, y, corr, path, top=10):
    names = list(corr.abs().sort_values(ascending=False).index[:top])
    idx = [feature_cols.index(n) for n in names]
    data = np.column_stack([y] + [X[:, i] for i in idx])
    labels = ["int.rate"] + names
    m = np.corrcoef(data, rowvar=False)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(m, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(labels)):          # label only the strong cells, not every cell
        for j in range(len(labels)):
            if abs(m[i, j]) >= 0.5 and i != j:
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    ax.set_title("How the top features relate to each other and to int.rate")
    _save(fig, path)


def plot_top_features_scatter(X, feature_cols, y, names, path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, name in zip(axes.ravel(), names):
        x = X[:, feature_cols.index(name)]
        ax.scatter(x, y * 100, s=6, alpha=0.18, color=BLUE, linewidths=0)
        slope, intercept = np.polyfit(x, y * 100, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, slope * xs + intercept, color=INK, linewidth=1.8)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{name}   (r = {r:+.2f})")
        ax.set_xlabel(name)
        ax.set_ylabel("int.rate (%)")
    fig.suptitle("Interest rate against the four most correlated numeric features", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    _save(fig, path)


def plot_rate_by_purpose(df, path):
    m = (df.groupby("purpose")["int.rate"].mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(m.index, m.values, color=BLUE, height=0.55)
    for name, v in m.items():
        ax.text(v + 0.1, name, f"{v:.2f}%", va="center", color=INK)
    ax.set_xlim(0, m.max() * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Average interest rate (%)")
    ax.set_title("Average interest rate by loan purpose")
    _save(fig, path)


# ----------------------------------------------------------------------------------------------
# 02  LOSS
# ----------------------------------------------------------------------------------------------
def plot_loss_curves(histories, path):
    """histories: {model name: list of training MSE per iteration}. Small multiples (each model has its own x-axis)."""
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5))
    for ax, (name, h) in zip(axes.ravel(), histories.items()):
        ax.plot(np.arange(1, len(h) + 1), h, color=MODEL_COLORS[name])
        ax.set_yscale("log")
        ax.set_title(SHORT[name])
        ax.set_xlabel("Boosting round" if "Boosting" in name else "Gradient-descent iteration")
        ax.set_ylabel("Training loss (MSE, log scale)")
        ax.plot([len(h)], [h[-1]], "o", color=MODEL_COLORS[name], markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"final {h[-1]:.2e}", (len(h), h[-1]), textcoords="offset points", xytext=(-6, -16), ha="right", color=INK_2)
    fig.suptitle("Training loss falls, then levels off - the models have finished learning", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    _save(fig, path)


def plot_train_vs_test_loss(loss_table, path):
    """loss_table: DataFrame with columns Model, Train loss (MSE), Test loss (MSE). Light bar = train, solid = test."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    models = list(MODEL_COLORS)
    x = np.arange(len(models))
    w = 0.34
    tr = [loss_table.loc[loss_table["Model"] == m, "Train loss (MSE)"].iloc[0] for m in models]
    te = [loss_table.loc[loss_table["Model"] == m, "Test loss (MSE)"].iloc[0] for m in models]
    for i, m in enumerate(models):
        ax.bar(x[i] - w / 2 - 0.01, tr[i], w, color=_tint(MODEL_COLORS[m]))
        ax.bar(x[i] + w / 2 + 0.01, te[i], w, color=MODEL_COLORS[m])
        ax.text(x[i] - w / 2, tr[i], f"{tr[i]:.2e}".replace("e-0", "e-"), ha="center", va="bottom", color=INK_2, fontsize=8)
        ax.text(x[i] + w / 2, te[i], f"{te[i]:.2e}".replace("e-0", "e-"), ha="center", va="bottom", color=INK, fontsize=8)
    ax.set_xticks(x, [SHORT[m] for m in models])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Train vs test loss (light bar = train, solid bar = test)")
    _save(fig, path)


# ----------------------------------------------------------------------------------------------
# 07  PERFORMANCE CURVE
# ----------------------------------------------------------------------------------------------
def plot_boosting_performance(n_trees, train_mse, test_mse, test_r2, path):
    color = MODEL_COLORS["4. Gradient Boosting (better)"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3))
    a.plot(n_trees, test_r2, color=color)
    peak = int(np.argmax(test_r2))
    a.plot([peak], [test_r2[peak]], "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    a.annotate(f"peak R2 {test_r2[peak]:.4f}\nat {peak} trees", (peak, test_r2[peak]), textcoords="offset points",
               xytext=(8, -34), color=INK_2)
    a.set_ylim(max(0, min(test_r2[20:]) - 0.05), max(test_r2) + 0.02)
    a.set_xlabel("Number of trees")
    a.set_ylabel("Test R2")
    a.set_title("Test R2 as trees are added")
    b.plot(n_trees, train_mse, color=NEUTRAL)
    b.plot(n_trees, test_mse, color=color)
    b.annotate("train", (n_trees[-1], train_mse[-1]), textcoords="offset points", xytext=(-4, 8), ha="right", color=INK_2)
    b.annotate("test", (n_trees[-1], test_mse[-1]), textcoords="offset points", xytext=(-4, 8), ha="right", color=INK_2)
    b.set_yscale("log")
    b.set_xlabel("Number of trees")
    b.set_ylabel("Loss (MSE, log scale)")
    b.set_title("Train vs test loss as trees are added")
    fig.tight_layout()
    _save(fig, path)


# ----------------------------------------------------------------------------------------------
# 08  R-SQUARE
# ----------------------------------------------------------------------------------------------
def plot_r2_comparison(table, path):
    """table: DataFrame indexed by model with 'Test R2', 'CV R2 mean', 'CV R2 std'."""
    models = list(MODEL_COLORS)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(models))
    vals = [table.loc[m, "Test R2"] for m in models]
    ax.barh(y, vals, color=[MODEL_COLORS[m] for m in models], height=0.5)
    cv = [table.loc[m, "CV R2 mean"] for m in models]
    sd = [table.loc[m, "CV R2 std"] for m in models]
    ax.errorbar(cv, y, xerr=sd, fmt="o", color=INK, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
                capsize=4, linewidth=1.4, label="5-fold CV mean +/- std")
    taught = max(table.loc[m, "Test R2"] for m in list(MODEL_COLORS)[:3])
    ax.axvline(taught, color=NEUTRAL, linewidth=1.2)
    ax.text(taught + 0.01, -0.62, "best taught model", color=INK_2, ha="left", va="center", fontsize=8)
    for yi, v in zip(y, vals):
        ax.text(0.01, yi, f"{v:.3f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, [SHORT[m] for m in models])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, len(models) - 0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("R2  (bar = test set)")
    ax.set_title("R2 of every model - the better model leads the taught ones")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    _save(fig, path)


def plot_actual_vs_predicted(y_true, predictions, r2_by_model, path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    lo, hi = y_true.min() * 100, y_true.max() * 100
    for ax, (name, pred) in zip(axes.ravel(), predictions.items()):
        ax.scatter(y_true * 100, pred * 100, s=8, alpha=0.35, color=MODEL_COLORS[name], linewidths=0)
        ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(f"{SHORT[name]}   (R2 = {r2_by_model[name]:.3f})")
        ax.set_xlabel("Actual int.rate (%)")
        ax.set_ylabel("Predicted int.rate (%)")
    fig.suptitle("Actual vs predicted on the test set - closer to the line is better", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    _save(fig, path)


def plot_cv_per_fold(cv_scores, path):
    models = list(MODEL_COLORS)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    rng = np.random.RandomState(0)
    for i, m in enumerate(models):
        v = np.array(cv_scores[m])
        ax.hlines(v.mean(), i - 0.28, i + 0.28, color=INK, linewidth=2, zorder=2)
        ax.plot(i + rng.uniform(-0.12, 0.12, len(v)), v, "o", color=MODEL_COLORS[m], markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.text(i + 0.32, v.mean(), f"{v.mean():.3f}", va="center", color=INK, fontsize=9)
    ax.set_xticks(range(len(models)), [SHORT[m] for m in models])
    ax.set_xlim(-0.5, len(models) - 0.2)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Test R2 in each fold")
    ax.set_title("5 train/test runs per model (dots = folds, bar = mean)")
    _save(fig, path)


# ----------------------------------------------------------------------------------------------
# 09  OTHERS (analysis)
# ----------------------------------------------------------------------------------------------
def plot_feature_importance(gain, perm, path, top=10):
    color = MODEL_COLORS["4. Gradient Boosting (better)"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, s, title, fmt in ((a, gain, "Share of total error reduction (tree splits)", "{:.0%}"),
                              (b, perm, "Rise in test loss when the feature is shuffled", "{:.1e}")):
        s = s.sort_values(ascending=False).head(top)[::-1]
        ax.barh(s.index, s.values, color=color, height=0.55)
        for name, v in list(s.items())[-3:]:
            ax.text(v, name, " " + fmt.format(v), va="center", color=INK)
        ax.set_xlim(0, s.max() * 1.2)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    fig.suptitle("Feature importance of the Gradient Boosting model - FICO dominates", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    _save(fig, path)


def plot_residuals(y_pred_by_model, y_true, path):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))
    for ax, (name, pred) in zip(axes.ravel(), y_pred_by_model.items()):
        ax.scatter(pred * 100, (y_true - pred) * 100, s=8, alpha=0.3, color=MODEL_COLORS[name], linewidths=0)
        ax.axhline(0, color=INK_2, linewidth=1.2)
        ax.set_title(SHORT[name])
        ax.set_xlabel("Predicted int.rate (%)")
        ax.set_ylabel("Residual = actual - predicted (pts)")
    fig.suptitle("Residuals should scatter randomly around 0 - patterns mean something is missed", x=0.01, ha="left",
                 fontweight="semibold")
    fig.tight_layout()
    _save(fig, path)


def plot_scratch_vs_sklearn(ver, path):
    """ver: DataFrame indexed by model with 'Scratch R2' and 'scikit-learn R2'. Points on the diagonal = identical."""
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    lo = float(min(ver["Scratch R2"].min(), ver["scikit-learn R2"].min())) - 0.03
    hi = float(max(ver["Scratch R2"].max(), ver["scikit-learn R2"].max())) + 0.03
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1.2)
    for name, row in ver.iterrows():
        ax.plot(row["scikit-learn R2"], row["Scratch R2"], "o", markersize=11, color=MODEL_COLORS[name],
                markeredgecolor=SURFACE, markeredgewidth=2, label=SHORT[name])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("scikit-learn R2 (ready-to-use function)")
    ax.set_ylabel("Our from-scratch R2")
    ax.set_title("Scratch models match scikit-learn (on the line = identical)")
    ax.legend(loc="upper left")
    _save(fig, path)
