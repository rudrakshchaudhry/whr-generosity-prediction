"""Feature and target definitions (unchanged from the original analysis)."""

from __future__ import annotations

import pandas as pd

FEATURES: list[str] = [
    "Log GDP per capita",
    "Social support",
    "Healthy life expectancy at birth",
    "Freedom to make life choices",
    "Perceptions of corruption",
    "Positive affect",
    "Negative affect",
]
TARGET = "Generosity"
GROUP = "country"


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return features ``X`` (NaNs kept), target ``y`` and country ``groups``.

    Rows with a missing target are dropped (they cannot be scored). Missing
    feature values are deliberately left as NaN: imputation happens inside the
    model pipeline so it is fitted on training folds only.
    """
    missing = [c for c in [*FEATURES, TARGET, GROUP] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    data = df.loc[df[TARGET].notna()].reset_index(drop=True)
    return data[FEATURES].copy(), data[TARGET].copy(), data[GROUP].copy()
