"""Evaluation protocols.

Two things live here:

* ``reproduce_original`` re-runs the 2018 notebook's protocol verbatim
  (global mean imputation and scaling, one random 75/25 row split) so the
  original numbers can be checked.
* ``nested_cv`` runs repeated nested cross-validation under an outer
  splitting scheme: ``random`` (shuffled K-fold over country-year rows, the
  default, matching the original analysis) or ``grouped`` (each country kept
  within a single fold; opt-in via ``--grouping country``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, GROUP, TARGET
from .models import ModelSpec

SCHEMES = ("grouped", "random")


def make_splitter(scheme: str, n_splits: int, seed: int):
    """Return a CV splitter. ``grouped`` keeps each country in a single fold."""
    if scheme == "grouped":
        return GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if scheme == "random":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raise ValueError(f"Unknown scheme {scheme!r}; expected one of {SCHEMES}")


def reproduce_original(df: pd.DataFrame) -> dict:
    """Re-run the original notebook's protocol exactly.

    Mean imputation and standard scaling are fitted on the full dataset before
    a random 75/25 split (``random_state=11``); models are an unregularised
    linear regression and a default random forest (``random_state=0``).
    """
    data = df[[*FEATURES, TARGET, GROUP]].dropna(subset=[TARGET]).copy()
    data[FEATURES] = data[FEATURES].fillna(data[FEATURES].mean())
    X = StandardScaler().fit_transform(data[FEATURES])
    y = data[TARGET].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=11)
    out: dict = {}
    for name, model in {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(random_state=0),
    }.items():
        pred = model.fit(X_tr, y_tr).predict(X_te)
        out[name] = {
            "r2": float(r2_score(y_te, pred)),
            "mse": float(mean_squared_error(y_te, pred)),
        }
    out["n_test_rows"] = int(len(y_te))
    return out


@dataclass
class FoldResult:
    r2: float
    mse: float
    mae: float
    best_params: dict
    estimator: object = field(repr=False)
    test_idx: np.ndarray = field(repr=False)
    repeat: int = 0


def _fit_tuned(
    spec: ModelSpec, X, y, groups, scheme: str, n_inner: int, seed: int, n_jobs: int = -1
):
    """Fit ``spec`` on training data, tuning on an inner CV of the same scheme."""
    if not spec.param_grid:
        return clone(spec.pipeline).fit(X, y), {}
    search = GridSearchCV(
        clone(spec.pipeline),
        spec.param_grid,
        cv=make_splitter(scheme, n_inner, seed),
        scoring="r2",
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X, y, groups=groups if scheme == "grouped" else None)
    return search.best_estimator_, {k: _py(v) for k, v in search.best_params_.items()}


def nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    specs: dict[str, ModelSpec],
    scheme: str,
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 0,
    n_repeats: int = 1,
) -> dict[str, list[FoldResult]]:
    """Repeated nested CV: outer folds estimate performance, inner folds tune.

    Repeat ``r`` uses ``seed + r`` for both the outer and inner fold
    assignment, so results do not hinge on one particular partition of the
    160 countries.
    """
    results: dict[str, list[FoldResult]] = {name: [] for name in specs}
    for r in range(n_repeats):
        s = seed + r
        outer = make_splitter(scheme, n_outer, s)
        split_groups = groups if scheme == "grouped" else None
        for train_idx, test_idx in outer.split(X, y, split_groups):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            g_tr = groups.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
            for name, spec in specs.items():
                est, params = _fit_tuned(spec, X_tr, y_tr, g_tr, scheme, n_inner, s)
                pred = est.predict(X_te)
                results[name].append(
                    FoldResult(
                        r2=float(r2_score(y_te, pred)),
                        mse=float(mean_squared_error(y_te, pred)),
                        mae=float(mean_absolute_error(y_te, pred)),
                        best_params=params,
                        estimator=est,
                        test_idx=test_idx,
                        repeat=r,
                    )
                )
    return results


def summarize(folds: list[FoldResult]) -> dict:
    """Mean and sample standard deviation (ddof=1) across all outer folds.

    Also reports the mean R^2 of each repeat, whose spread shows how much the
    estimate depends on the particular fold assignment.
    """
    out: dict = {}
    for metric in ("r2", "mse", "mae"):
        vals = np.array([getattr(f, metric) for f in folds])
        out[f"{metric}_mean"] = float(vals.mean())
        out[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out[f"{metric}_folds"] = [float(v) for v in vals]
    repeats = sorted({f.repeat for f in folds})
    out["r2_repeat_means"] = [
        float(np.mean([f.r2 for f in folds if f.repeat == r])) for r in repeats
    ]
    out["best_params_per_fold"] = [f.best_params for f in folds]
    return out


def heldout_permutation_importance(
    folds: list[FoldResult],
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Permutation importance (drop in R^2) on each outer *test* fold.

    Each fold's model never saw that fold's rows, so importances are measured
    on held-out data. Returned as mean and std across folds of the per-fold
    mean importance.
    """
    per_fold = []
    for i, f in enumerate(folds):
        r = permutation_importance(
            f.estimator,
            X.iloc[f.test_idx],
            y.iloc[f.test_idx],
            scoring="r2",
            n_repeats=n_repeats,
            random_state=seed + i,
            n_jobs=1,
        )
        per_fold.append(r.importances_mean)
    arr = np.vstack(per_fold)
    return {
        feat: {"mean": float(arr[:, j].mean()), "std": float(arr[:, j].std(ddof=1))}
        for j, feat in enumerate(X.columns)
    }


def fit_final(spec: ModelSpec, X, y, groups, scheme: str, n_inner: int, seed: int):
    """Tune with an inner CV of ``scheme`` on all rows and refit (for plots)."""
    return _fit_tuned(spec, X, y, groups, scheme, n_inner, seed)


def between_country_variance_share(y: pd.Series, groups: pd.Series) -> float:
    """Share of target variance explained by country means (eta squared)."""
    total = float(((y - y.mean()) ** 2).sum())
    within = float(((y - y.groupby(groups).transform("mean")) ** 2).sum())
    return 1.0 - within / total


def _py(v):
    """Convert numpy scalars to plain Python for JSON."""
    if isinstance(v, np.generic):
        return v.item()
    return v
