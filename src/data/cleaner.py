"""Data cleaning utilities for FoodVision AI."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.utils.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.utils.logger import get_logger


logger = get_logger(__name__)


def standardize_city_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize city names to title case.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with standardized city names.
    """
    if "ciudad" not in df.columns:
        logger.warning("Column ciudad not found")
        return df

    before = df["ciudad"].copy()
    df["ciudad"] = df["ciudad"].astype(str).str.lower().str.title()
    changed = (before != df["ciudad"]).sum()
    logger.info(f"Standardized city names: {changed} values updated")
    return df


def fix_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace negative values in selected columns with NaN.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with negative values replaced.
    """
    columns = ["tiempo_preparacion", "valor_total"]
    for column in columns:
        if column in df.columns:
            mask = df[column] < 0
            count = int(mask.sum())
            df.loc[mask, column] = np.nan
            logger.info(f"Negative values in {column} set to NaN: {count}")
        else:
            logger.warning(f"Column {column} not found")
    return df


def fix_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap outliers for preparation time.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with outliers capped.
    """
    column = "tiempo_preparacion"
    if column not in df.columns:
        logger.warning("Column tiempo_preparacion not found")
        return df

    before = df[column].copy()
    df[column] = df[column].clip(lower=0, upper=120)
    changed = (before != df[column]).sum()
    logger.info(f"Capped outliers in {column}: {changed} values updated")
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values for numeric and categorical columns.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with imputed values.
    """
    numeric_cols = [col for col in NUMERIC_COLUMNS if col in df.columns]
    categorical_cols = [col for col in CATEGORICAL_COLUMNS if col in df.columns]

    if numeric_cols:
        imputer_num = SimpleImputer(strategy="median")
        df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
        logger.info("Imputed missing numeric values with median")
    else:
        logger.warning("No numeric columns available for imputation")

    if categorical_cols:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
        logger.info("Imputed missing categorical values with most frequent")
    else:
        logger.warning("No categorical columns available for imputation")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset without duplicate rows.
    """
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    removed = before - after
    logger.info(f"Removed duplicate rows: {removed}")
    return df


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and correct ranges for customer ratings.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with validated ranges.
    """
    column = "calificacion_cliente"
    if column not in df.columns:
        logger.warning("Column calificacion_cliente not found")
        return df

    mask = (df[column] < 1) | (df[column] > 5)
    count = int(mask.sum())
    df.loc[mask, column] = np.nan
    logger.info(f"Out-of-range values in {column} set to NaN: {count}")

    imputer = SimpleImputer(strategy="median")
    df[[column]] = imputer.fit_transform(df[[column]])
    logger.info("Imputed calificacion_cliente with median")
    return df


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Run the full data cleaning pipeline.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        Tuple[pd.DataFrame, Dict[str, int]]: Cleaned dataset and cleaning report.
    """
    report: Dict[str, int] = {}

    df_clean = df.copy()

    df_clean = standardize_city_names(df_clean)
    df_clean = fix_negative_values(df_clean)
    df_clean = fix_outliers(df_clean)

    before_dup = df_clean.shape[0]
    df_clean = remove_duplicates(df_clean)
    report["duplicates_removed"] = before_dup - df_clean.shape[0]

    before_missing = int(df_clean.isna().sum().sum())
    df_clean = validate_ranges(df_clean)
    df_clean = impute_missing_values(df_clean)
    after_missing = int(df_clean.isna().sum().sum())
    report["missing_values_before"] = before_missing
    report["missing_values_after"] = after_missing

    if df_clean.empty:
        message = "Empty dataset after cleaning"
        logger.error(message)
        raise ValueError(message)

    logger.info("Dataset cleaning completed")
    return df_clean, report
