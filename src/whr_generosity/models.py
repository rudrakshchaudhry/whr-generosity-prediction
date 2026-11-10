"""Model pipelines and their hyperparameter search spaces.

Every model is an sklearn ``Pipeline`` so that imputation and scaling are
fitted only on whatever data ``fit`` receives (a training fold), never on the
evaluation fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline
    param_grid: dict[str, list] = field(default_factory=dict)


def _imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median")


def build_models(seed: int = 0, quick: bool = False) -> dict[str, ModelSpec]:
    """Return the model zoo keyed by short name.

    ``quick`` shrinks forests and search grids for tests and CI smoke runs.
    """
    n_trees = 50 if quick else 300

    dummy = Pipeline([("model", DummyRegressor(strategy="mean"))])

    # Ridge: alpha is chosen by the same inner CV as the other models (rather
    # than RidgeCV's built-in leave-one-out) so every model is tuned the same
    # way, including under --grouping country.
    ridge = Pipeline(
        [
            ("impute", _imputer()),
            ("scale", StandardScaler()),
            ("model", Ridge()),
        ]
    )
    ridge_grid = {"model__alpha": list(np.logspace(-3, 3, 4 if quick else 13))}

    rf = Pipeline(
        [
            ("impute", _imputer()),
            (
                "model",
                RandomForestRegressor(n_estimators=n_trees, random_state=seed, n_jobs=1),
            ),
        ]
    )
    rf_grid = (
        {"model__min_samples_leaf": [1, 10]}
        if quick
        else {
            "model__max_depth": [None, 8],
            "model__min_samples_leaf": [1, 5, 20],
            "model__max_features": [1.0, 0.5],
        }
    )

    # HistGradientBoosting handles NaN natively, so no imputer is needed:
    # missingness is learned from training folds only.
    hgb = Pipeline(
        [
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=100 if quick else 300,
                    early_stopping=False,
                    random_state=seed,
                ),
            ),
        ]
    )
    hgb_grid = (
        {"model__learning_rate": [0.1]}
        if quick
        else {
            "model__learning_rate": [0.03, 0.1],
            "model__max_leaf_nodes": [7, 31],
            "model__min_samples_leaf": [10, 40],
            "model__l2_regularization": [0.0, 1.0],
        }
    )

    return {
        "dummy_mean": ModelSpec("dummy_mean", dummy, {}),
        "ridge": ModelSpec("ridge", ridge, ridge_grid),
        "random_forest": ModelSpec("random_forest", rf, rf_grid),
        "hist_gradient_boosting": ModelSpec("hist_gradient_boosting", hgb, hgb_grid),
    }
