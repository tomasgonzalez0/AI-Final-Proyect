"""Model comparison utilities for FoodVision AI."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


def compare_all_models(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare models based on their evaluation metrics.

    Args:
        results_list (List[Dict[str, Any]]): List of evaluation result dictionaries.

    Returns:
        pd.DataFrame: Sorted comparison DataFrame.
    """
    if not results_list:
        logger.warning("No results to compare")
        return pd.DataFrame()

    df = pd.DataFrame(results_list)

    if "task" not in df.columns:
        logger.warning("Task column missing in results")
        return df

    task = df["task"].iloc[0]
    if task == "classification" and "accuracy" in df.columns:
        df = df.sort_values(by="accuracy", ascending=False)
    elif task == "regression" and "r2" in df.columns:
        df = df.sort_values(by="r2", ascending=False)
    else:
        logger.warning("No valid metric found for sorting")

    logger.info("Model comparison completed")
    return df


def save_comparison_report(df: pd.DataFrame, filepath: Any) -> None:
    """
    Save model comparison report to CSV.

    Args:
        df (pd.DataFrame): Comparison DataFrame.
        filepath (Any): File path for the CSV.

    Returns:
        None: This function returns None.
    """
    try:
        df.to_csv(filepath, index=False)
        logger.info(f"Saved comparison report to {filepath}")
    except OSError as exc:
        logger.error("Failed to save comparison report")
        raise OSError("Failed to save comparison report") from exc
