"""Model evaluation utilities for FoodVision AI."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from src.utils.logger import get_logger


logger = get_logger(__name__)


def evaluate_classification(
    y_true: Any, y_pred: Any, model_name: str
) -> Dict[str, Any]:
    """
    Evaluate classification predictions.

    Args:
        y_true (Any): True labels.
        y_pred (Any): Predicted labels.
        model_name (str): Model name.

    Returns:
        Dict[str, Any]: Dictionary with evaluation metrics.
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_val = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    matrix = confusion_matrix(y_true, y_pred)
    print(matrix)
    logger.info(f"Confusion matrix for {model_name} printed")

    results = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_val,
        "task": "classification",
    }
    print(results)
    return results


def evaluate_regression(y_true: Any, y_pred: Any, model_name: str) -> Dict[str, Any]:
    """
    Evaluate regression predictions.

    Args:
        y_true (Any): True values.
        y_pred (Any): Predicted values.
        model_name (str): Model name.

    Returns:
        Dict[str, Any]: Dictionary with evaluation metrics.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    results = {
        "model_name": model_name,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "task": "regression",
    }
    print(results)
    return results


def print_metrics_table(results_list: List[Dict[str, Any]]) -> None:
    """
    Print metrics in a formatted table.

    Args:
        results_list (List[Dict[str, Any]]): List of metric dictionaries.

    Returns:
        None: This function returns None.
    """
    if not results_list:
        print("No results to display")
        return

    keys = list(results_list[0].keys())
    col_widths = {key: len(key) for key in keys}

    for result in results_list:
        for key in keys:
            value = result.get(key, "")
            col_widths[key] = max(col_widths[key], len(str(value)))

    header = " | ".join(key.ljust(col_widths[key]) for key in keys)
    separator = "-+-".join("-" * col_widths[key] for key in keys)
    print(header)
    print(separator)

    for result in results_list:
        row = " | ".join(str(result.get(key, "")).ljust(col_widths[key]) for key in keys)
        print(row)
