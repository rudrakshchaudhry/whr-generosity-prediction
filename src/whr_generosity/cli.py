"""Command-line entry point: regenerate results/ and figures/ from scratch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sklearn

from . import plots
from .data import load_data
from .evaluate import (
    between_country_variance_share,
    fit_final,
    heldout_permutation_importance,
    nested_cv,
    reproduce_original,
    summarize,
)
from .features import FEATURES, GROUP, TARGET, make_xy
from .models import build_models
from .report import render_markdown

N_PDP_FEATURES = 3


def _round(obj, digits: int = 6):
    """Round floats recursively so the JSON is stable and readable."""
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, digits) for v in obj]
    return obj


def run(
    data_path: str | Path | None = None,
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
    seed: int = 0,
    n_outer: int = 5,
    n_inner: int = 5,
    n_repeats: int = 5,
    quick: bool = False,
    make_figures: bool = True,
    grouping: str = "none",
) -> dict:
    """Run the study and write metrics.json, metrics.md and figures.

    By default the outer CV is shuffled K-fold over rows (``random``).
    ``grouping="country"`` additionally runs the same procedure with folds
    grouped by country and uses it for importances and partial dependence.
    """
    if grouping not in ("none", "country"):
        raise ValueError(f"grouping must be 'none' or 'country', got {grouping!r}")
    schemes = ["random"] + (["grouped"] if grouping == "country" else [])
    primary = schemes[-1]
    results_dir, figures_dir = Path(results_dir), Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    X, y, groups = make_xy(df)
    specs = build_models(seed=seed, quick=quick)

    cv: dict = {}
    folds_by_scheme: dict = {}
    for scheme in schemes:
        folds = nested_cv(X, y, groups, specs, scheme, n_outer, n_inner, seed, n_repeats)
        folds_by_scheme[scheme] = folds
        cv[scheme] = {name: summarize(f) for name, f in folds.items()}

    candidates = [m for m in cv[primary] if m != "dummy_mean"]
    best = max(candidates, key=lambda m: cv[primary][m]["r2_mean"])
    pi = {
        m: heldout_permutation_importance(
            folds_by_scheme[primary][m], X, y, n_repeats=5 if quick else 10, seed=seed
        )
        for m in candidates
    }
    top = sorted(pi[best], key=lambda f: pi[best][f]["mean"], reverse=True)
    pdp_features = top[:N_PDP_FEATURES]

    years = df.loc[df[TARGET].notna(), "year"]
    metrics = {
        "config": {
            "seed": seed,
            "n_outer_splits": n_outer,
            "n_inner_splits": n_inner,
            "n_repeats": n_repeats,
            "quick": quick,
            "grouping": grouping,
            "primary_scheme": primary,
            "features": FEATURES,
            "target": TARGET,
            "group": GROUP,
            "std": "sample standard deviation (ddof=1) across all outer folds of all repeats",
            "sklearn_version": sklearn.__version__,
        },
        "data_summary": {
            "n_rows_raw": int(len(df)),
            "n_rows": int(len(y)),
            "n_countries": int(groups.nunique()),
            "year_min": int(years.min()),
            "year_max": int(years.max()),
            "missing_feature_values": {f: int(X[f].isna().sum()) for f in FEATURES},
        },
        "original_protocol": reproduce_original(df),
        "cv": cv,
        "best_model": best,
        "permutation_importance": pi,
        "pdp_features": pdp_features,
    }
    if grouping == "country":
        metrics["data_summary"]["between_country_variance_share"] = between_country_variance_share(
            y, groups
        )
    metrics = _round(metrics)

    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (results_dir / "metrics.md").write_text(render_markdown(metrics), encoding="utf-8")

    if make_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)
        plots.model_comparison(
            cv, figures_dir / "model_comparison.png", n_folds=n_outer * n_repeats
        )
        plots.permutation_importance_plot(
            pi[best], best, figures_dir / "permutation_importance.png"
        )
        final, _ = fit_final(specs[best], X, y, groups, primary, n_inner, seed)
        plots.partial_dependence_plot(
            final, X, pdp_features, best, figures_dir / "partial_dependence.png", seed
        )
    return metrics


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="whr-generosity",
        description="Reproduce the WHR 2018 Generosity prediction study.",
    )
    p.add_argument("--data", default=None, help="Path to WHR2018Chapter2OnlineData.csv")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--figures-dir", default="figures")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outer-splits", type=int, default=5)
    p.add_argument("--inner-splits", type=int, default=5)
    p.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Repetitions of the outer CV with different fold seeds.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Small grids and forests (smoke test / CI); not for reporting.",
    )
    p.add_argument("--no-figures", action="store_true")
    p.add_argument(
        "--grouping",
        choices=["none", "country"],
        default="none",
        help="'country' also evaluates with folds grouped by country (off by default).",
    )
    a = p.parse_args(argv)
    m = run(
        data_path=a.data,
        results_dir=a.results_dir,
        figures_dir=a.figures_dir,
        seed=a.seed,
        n_outer=a.outer_splits,
        n_inner=a.inner_splits,
        n_repeats=a.repeats,
        quick=a.quick,
        make_figures=not a.no_figures,
        grouping=a.grouping,
    )
    g = m["cv"][m["config"]["primary_scheme"]][m["best_model"]]
    print(
        f"Best model ({m['config']['primary_scheme']} CV): {m['best_model']} "
        f"R2 = {g['r2_mean']:.3f} +/- {g['r2_std']:.3f}"
    )
    print(f"Wrote {Path(a.results_dir) / 'metrics.json'}")
    return 0
