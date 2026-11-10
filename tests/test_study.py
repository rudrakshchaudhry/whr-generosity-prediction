import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from whr_generosity.cli import main
from whr_generosity.data import EXPECTED_SHAPE, load_data
from whr_generosity.evaluate import make_splitter, nested_cv, reproduce_original
from whr_generosity.features import FEATURES, TARGET, make_xy
from whr_generosity.models import build_models
from whr_generosity.report import cv_table, importance_table, original_table

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def df():
    return load_data()


@pytest.fixture(scope="module")
def xyg(df):
    return make_xy(df)


# --- data ------------------------------------------------------------------


def test_loader_shape_and_columns(df):
    assert df.shape == EXPECTED_SHAPE
    for col in ["country", "year", TARGET, *FEATURES]:
        assert col in df.columns
    assert not df.duplicated(["country", "year"]).any()


def test_make_xy_drops_missing_target_keeps_feature_nans(df, xyg):
    X, y, g = xyg
    assert list(X.columns) == FEATURES
    assert len(X) == len(y) == len(g) == df[TARGET].notna().sum() == 1482
    assert y.notna().all()
    assert X.isna().any().any(), "feature NaNs must reach the pipeline, not be pre-filled"


# --- splitting -------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_grouped_split_has_zero_country_overlap(xyg, seed):
    X, y, g = xyg
    splitter = make_splitter("grouped", 5, seed)
    seen_test = set()
    for tr, te in splitter.split(X, y, g):
        train_c, test_c = set(g.iloc[tr]), set(g.iloc[te])
        assert train_c.isdisjoint(test_c)
        seen_test |= test_c
    assert seen_test == set(g)  # every country is evaluated exactly once


def test_inner_tuning_is_grouped_too(xyg, monkeypatch):
    """GridSearchCV inside each outer fold must receive the country groups."""
    X, y, g = xyg
    calls = []
    orig = GroupKFold.split

    def spy(self, X_, y_=None, groups=None):
        calls.append(groups)
        return orig(self, X_, y_, groups)

    monkeypatch.setattr(GroupKFold, "split", spy)
    specs = {"ridge": build_models(quick=True)["ridge"]}
    nested_cv(X, y, g, specs, "grouped", n_outer=2, n_inner=2, seed=0)
    assert len(calls) >= 3  # outer split + at least one inner search per fold
    assert all(c is not None for c in calls)


# --- leakage in preprocessing ---------------------------------------------


def test_imputer_and_scaler_fitted_on_training_fold_only(xyg):
    X, y, g = xyg
    specs = {"ridge": build_models(quick=True)["ridge"]}
    folds = nested_cv(X, y, g, specs, "grouped", n_outer=3, n_inner=2, seed=0)["ridge"]
    for f in folds:
        train_mask = np.ones(len(X), dtype=bool)
        train_mask[f.test_idx] = False
        X_train = X[train_mask]
        imputer = f.estimator.named_steps["impute"]
        scaler = f.estimator.named_steps["scale"]
        np.testing.assert_allclose(imputer.statistics_, X_train.median().to_numpy())
        # Scaler sees training rows after imputation
        X_imp = X_train.fillna(X_train.median())
        np.testing.assert_allclose(scaler.mean_, X_imp.mean().to_numpy())
        assert scaler.n_samples_seen_ == train_mask.sum()
        # ...and the full-data statistics are different (so the test has teeth)
        assert not np.allclose(imputer.statistics_, X.median().to_numpy())


def test_pipeline_handles_nan_and_test_fold_values_do_not_matter(xyg):
    """Changing evaluation-fold feature values must not change the fitted model."""
    X, y, g = xyg
    tr, te = next(make_splitter("grouped", 5, 0).split(X, y, g))
    pipe = build_models(quick=True)["ridge"].pipeline.set_params(model__alpha=1.0)
    X2 = X.copy()
    X2.iloc[te] = X2.iloc[te] * 1000  # corrupt the evaluation rows
    a = pipe.fit(X.iloc[tr], y.iloc[tr]).named_steps["model"].coef_.copy()
    b = pipe.fit(X2.iloc[tr], y.iloc[tr]).named_steps["model"].coef_.copy()
    np.testing.assert_array_equal(a, b)


# --- original protocol ----------------------------------------------------


def test_original_protocol_reproduces_readme(df):
    o = reproduce_original(df)
    assert round(o["linear_regression"]["r2"], 3) == 0.268
    assert round(o["linear_regression"]["mse"], 3) == 0.019
    assert round(o["random_forest"]["r2"], 2) == 0.61


# --- CLI ---------------------------------------------------------------------


EXPECTED_KEYS = {
    "config",
    "data_summary",
    "original_protocol",
    "cv",
    "best_model",
    "permutation_importance",
    "pdp_features",
}


QUICK = ["--quick", "--outer-splits", "2", "--inner-splits", "2", "--repeats", "1", "--no-figures"]
MODELS = {"dummy_mean", "ridge", "random_forest", "hist_gradient_boosting"}


def _check_scheme(scheme: dict, n_folds: int):
    assert set(scheme) == MODELS
    for s in scheme.values():
        assert {"r2_mean", "r2_std", "mse_mean", "mae_mean", "r2_folds", "r2_repeat_means"} <= set(
            s
        )
        assert len(s["r2_folds"]) == n_folds


def test_cli_writes_metrics_and_is_deterministic(tmp_path):
    assert main([*QUICK, "--results-dir", str(tmp_path / "a")]) == 0
    assert main([*QUICK, "--results-dir", str(tmp_path / "b")]) == 0
    a = (tmp_path / "a" / "metrics.json").read_text()
    b = (tmp_path / "b" / "metrics.json").read_text()
    assert a == b

    m = json.loads(a)
    assert set(m) == EXPECTED_KEYS
    assert set(m["cv"]) == {"random"}  # default run: random CV only
    assert m["config"]["primary_scheme"] == "random"
    _check_scheme(m["cv"]["random"], 2)
    assert m["best_model"] in MODELS - {"dummy_mean"}
    assert set(m["permutation_importance"][m["best_model"]]) == set(FEATURES)
    assert (tmp_path / "a" / "metrics.md").exists()


def test_cli_grouping_country_option(tmp_path):
    assert main([*QUICK, "--grouping", "country", "--results-dir", str(tmp_path)]) == 0
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert set(m["cv"]) == {"random", "grouped"}
    assert m["config"]["primary_scheme"] == "grouped"
    _check_scheme(m["cv"]["grouped"], 2)


# --- committed artefacts stay consistent ---------------------------------


def test_readme_matches_committed_metrics():
    metrics = json.loads((REPO / "results" / "metrics.json").read_text())
    assert metrics["config"]["quick"] is False
    assert metrics["config"]["grouping"] == "none"
    readme = (REPO / "README.md").read_text()
    for table in (cv_table, original_table, importance_table):
        assert table(metrics) in readme
    assert metrics["best_model"] == "hist_gradient_boosting"  # README text names it


def test_committed_figures_exist():
    for name in ["model_comparison.png", "permutation_importance.png", "partial_dependence.png"]:
        assert (REPO / "figures" / name).stat().st_size > 0
