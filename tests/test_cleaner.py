"""Tests for data cleaning functions."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from src.data.cleaner import clean_dataset
from src.data.loader import load_excel


DATA_PATH = "data/restaurante_foodvision.xlsx"


def _load_dataset() -> pd.DataFrame:
    """
    Load the real dataset for testing.

    Args:
        None: This function does not take arguments.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return load_excel(DATA_PATH)


def test_standardize_city_names() -> Tuple[bool, str]:
    """
    Validate city name standardization.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    df = _load_dataset()
    df_clean, _ = clean_dataset(df)
    cities = df_clean["ciudad"].dropna().astype(str)
    lower_mask = cities.str.islower()
    upper_mask = cities.str.isupper()
    offending = cities[lower_mask | upper_mask].unique().tolist()

    known_bad = {"bogota", "MEDELLIN", "medellín"}
    present_bad = [c for c in cities.unique() if c in known_bad]

    if offending or present_bad:
        reason = f"Found non-title case values: {offending or present_bad}"
        return False, reason

    unique_values = sorted(cities.unique().tolist())
    return True, f"Unique ciudad values: {unique_values}"


def test_fix_negative_values() -> Tuple[bool, str]:
    """
    Validate negative value fixes.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    df = _load_dataset()
    df.loc[0, "tiempo_preparacion"] = -15
    df.loc[1, "valor_total"] = -9000
    df_clean, _ = clean_dataset(df)

    negatives = {}
    for column in ["tiempo_preparacion", "valor_total"]:
        if (df_clean[column] < 0).any():
            negatives[column] = int((df_clean[column] < 0).sum())

    if negatives:
        return False, f"Negative values remain: {negatives}"
    return True, "No negative values remain"


def test_fix_outliers() -> Tuple[bool, str]:
    """
    Validate outlier capping for preparation time.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    df = _load_dataset()
    df.loc[0, "tiempo_preparacion"] = 350
    df_clean, _ = clean_dataset(df)
    max_val = float(df_clean["tiempo_preparacion"].max())
    if max_val <= 120:
        return True, f"Max tiempo_preparacion: {max_val}"
    return False, f"Outlier not capped, max found: {max_val}"


def test_impute_missing_values() -> Tuple[bool, str]:
    """
    Validate imputation of missing values.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    df = _load_dataset()
    df.loc[:4, "tiempo_preparacion"] = np.nan
    df.loc[:2, "tipo_comida"] = np.nan
    df_clean, _ = clean_dataset(df)

    missing_counts = {
        "tiempo_preparacion": int(df_clean["tiempo_preparacion"].isnull().sum()),
        "tipo_comida": int(df_clean["tipo_comida"].isnull().sum()),
    }

    if missing_counts["tiempo_preparacion"] == 0 and missing_counts["tipo_comida"] == 0:
        return True, "No missing values remain"
    return False, f"Missing values remain: {missing_counts}"


def test_remove_duplicates() -> Tuple[bool, str]:
    """
    Validate duplicate removal.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    df = _load_dataset()
    df_with_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df_clean, report = clean_dataset(df_with_dup)

    if len(df_clean) < len(df_with_dup) and report.get("duplicates_removed", 0) >= 1:
        return True, "Duplicate removed"
    reason = f"Row count before: {len(df_with_dup)} after: {len(df_clean)}"
    return False, reason


def test_validate_ranges() -> Tuple[bool, str]:
    """
    Validate rating range enforcement.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    df = _load_dataset()
    df.loc[0, "calificacion_cliente"] = 0
    df.loc[1, "calificacion_cliente"] = 7
    df_clean, _ = clean_dataset(df)

    in_range = df_clean["calificacion_cliente"].between(1, 5)
    if in_range.all():
        return True, "All ratings within range"
    bad_values = df_clean.loc[~in_range, "calificacion_cliente"].unique().tolist()
    return False, f"Out-of-range values remain: {bad_values}"


def test_cleaning_report_structure() -> Tuple[bool, str]:
    """
    Validate cleaning report structure.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    df = _load_dataset()
    _, report = clean_dataset(df)
    required_keys = {"duplicates_removed", "missing_values_before", "missing_values_after"}
    if not isinstance(report, dict):
        return False, "Cleaning report is not a dict"
    missing = sorted(required_keys - set(report.keys()))
    if missing:
        return False, f"Missing keys in report: {missing}"
    return True, "Cleaning report structure is valid"
