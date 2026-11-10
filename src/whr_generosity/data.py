"""Loading the World Happiness Report 2018 (Chapter 2) panel data."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DATA_FILENAME = "WHR2018Chapter2OnlineData.csv"
EXPECTED_SHAPE = (1562, 19)
ID_COLUMNS = ("country", "year")


def default_data_path() -> Path:
    """Locate the bundled CSV.

    Resolution order: the ``WHR_DATA`` environment variable, then
    ``<repo root>/data/<DATA_FILENAME>`` (works for a source checkout or an
    editable install).
    """
    env = os.environ.get("WHR_DATA")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / DATA_FILENAME


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Read the raw WHR 2018 panel (one row per country-year).

    The frame is sorted by country and year so downstream results do not
    depend on the row order of the file.
    """
    path = Path(path) if path is not None else default_data_path()
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path}. Pass --data or set WHR_DATA.")
    df = pd.read_csv(path)
    missing = [c for c in ID_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Data file is missing required columns: {missing}")
    return df.sort_values(list(ID_COLUMNS), kind="stable").reset_index(drop=True)
