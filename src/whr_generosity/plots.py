"""Static figures written to ``figures/``."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.inspection import PartialDependenceDisplay  # noqa: E402

# Validated categorical slots (orange for random CV, blue for grouped) and neutral inks.
GROUPED_COLOR = "#2a78d6"
RANDOM_COLOR = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"

MODEL_LABELS = {
    "dummy_mean": "Mean baseline",
    "ridge": "Ridge",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Hist. gradient boosting",
}

_RC = {
    "font.size": 10,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
}

# PNG metadata otherwise embeds the matplotlib version string only; drop
# "Software" too so figures are byte-stable across patch releases.
_PNG_META = {"Software": None}


SCHEME_STYLE = {
    "random": (RANDOM_COLOR, "5-fold random CV"),
    "grouped": (GROUPED_COLOR, "5-fold CV grouped by country"),
}


def model_comparison(cv: dict, path: Path, n_folds: int) -> None:
    """R^2 mean +/- std per model for each evaluated CV scheme."""
    schemes = [s for s in ("random", "grouped") if s in cv]
    models = list(cv[schemes[0]].keys())
    offsets = [0.0] if len(schemes) == 1 else [-0.15, 0.15]
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for offset, scheme in zip(offsets, schemes, strict=True):
            color, label = SCHEME_STYLE[scheme]
            means = [cv[scheme][m]["r2_mean"] for m in models]
            stds = [cv[scheme][m]["r2_std"] for m in models]
            ys = [i + offset for i in range(len(models))]
            ax.errorbar(
                means,
                ys,
                xerr=stds,
                fmt="o",
                color=color,
                ms=7,
                elinewidth=2,
                capsize=0,
                label=label,
            )
        ax.axvline(0, color=INK_MUTED, lw=1)
        ax.set_yticks(range(len(models)), [MODEL_LABELS.get(m, m) for m in models])
        ax.invert_yaxis()
        ax.set_xlabel(f"Out-of-fold R² (mean ± std over {n_folds} outer folds)")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        if len(schemes) > 1:
            ax.legend(frameon=False, loc="upper right")
        title = "Predicting Generosity: model comparison"
        if len(schemes) == 1:
            title += f" ({SCHEME_STYLE[schemes[0]][1]})"
        ax.set_title(title, loc="left", color=INK)
        fig.savefig(path, metadata=_PNG_META)
        plt.close(fig)


def permutation_importance_plot(pi: dict, model_name: str, path: Path) -> None:
    """Horizontal bars: mean drop in R^2 on held-out folds, +/- std."""
    items = sorted(pi.items(), key=lambda kv: kv[1]["mean"], reverse=True)
    names = [k for k, _ in items]
    means = [v["mean"] for _, v in items]
    stds = [v["std"] for _, v in items]
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.barh(
            names,
            means,
            xerr=stds,
            color=GROUPED_COLOR,
            height=0.6,
            error_kw={"ecolor": INK_MUTED, "elinewidth": 1.5},
        )
        ax.axvline(0, color=INK_MUTED, lw=1)
        ax.invert_yaxis()
        ax.set_xlabel("Drop in R² when the feature is permuted (held-out folds)")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(
            f"Permutation importance, {MODEL_LABELS.get(model_name, model_name)}",
            loc="left",
            color=INK,
        )
        fig.savefig(path, metadata=_PNG_META)
        plt.close(fig)


def partial_dependence_plot(
    estimator, X, features: list[str], model_name: str, path: Path, seed: int = 0
) -> None:
    """Average partial dependence with a sample of ICE curves."""
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(features), figsize=(3.4 * len(features), 3.2), sharey=True)
        disp = PartialDependenceDisplay.from_estimator(
            estimator,
            X,
            features,
            kind="both",
            subsample=80,
            random_state=seed,
            ax=axes,
            grid_resolution=40,
            ice_lines_kw={"color": GROUPED_COLOR, "alpha": 0.12, "lw": 0.8},
            pd_line_kw={"color": INK, "lw": 2, "linestyle": "-"},
        )
        for ax in disp.axes_.ravel():
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        fig.suptitle(
            f"Partial dependence of Generosity, {MODEL_LABELS.get(model_name, model_name)}"
            " (black: average; blue: individual rows)",
            x=0.01,
            ha="left",
            color=INK,
            fontsize=10,
        )
        fig.savefig(path, metadata=_PNG_META)
        plt.close(fig)
