"""Plotting utilities for FoodVision AI."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from src.utils.config import PLOTS_DIR
from src.utils.logger import get_logger


logger = get_logger(__name__)


def _import_plot_libs() -> Tuple[Any, Any]:
    """Import plotting libraries lazily.

    Returns:
        Tuple[Any, Any]: (plt, sns)

    Raises:
        ModuleNotFoundError: If matplotlib or seaborn is not installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        return plt, sns
    except ModuleNotFoundError as exc:
        message = (
            "Plotting dependencies are not installed. "
            "Install them with: python -m pip install -r requirements.txt"
        )
        logger.warning(message)
        raise ModuleNotFoundError(message) from exc


def plot_data_distribution(df: pd.DataFrame) -> None:
    """
    Plot histograms for numeric columns in the dataset.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        None: This function returns None.
    """
    try:
        plt, sns = _import_plot_libs()
    except ModuleNotFoundError:
        return

    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) == 0:
        logger.warning("No numeric columns for data distribution plot")
        return

    plt.figure(figsize=(12, 6))
    for column in numeric_cols:
        sns.histplot(df[column].dropna(), kde=True, label=column, color="tab:blue", alpha=0.4)

    plt.title("Data Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()

    file_path = PLOTS_DIR / "data_distribution.png"
    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(file_path, dpi=150)
        logger.info(f"Saved plot to {file_path}")
    except IOError as exc:
        logger.error("Failed to save data distribution plot")
        raise IOError("Failed to save data distribution plot") from exc
    finally:
        plt.close()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Plot a correlation heatmap for numeric columns.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        None: This function returns None.
    """
    try:
        plt, sns = _import_plot_libs()
    except ModuleNotFoundError:
        return

    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        logger.warning("No numeric columns for correlation heatmap")
        return

    corr = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Features")

    file_path = PLOTS_DIR / "correlation_heatmap.png"
    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(file_path, dpi=150)
        logger.info(f"Saved plot to {file_path}")
    except IOError as exc:
        logger.error("Failed to save correlation heatmap")
        raise IOError("Failed to save correlation heatmap") from exc
    finally:
        plt.close()


def plot_confusion_matrix(y_true: Any, y_pred: Any, model_name: str) -> None:
    """
    Plot confusion matrix for classification results.

    Args:
        y_true (Any): True labels.
        y_pred (Any): Predicted labels.
        model_name (str): Model name used in the plot title.

    Returns:
        None: This function returns None.
    """
    try:
        plt, sns = _import_plot_libs()
    except ModuleNotFoundError:
        return

    try:
        from sklearn.metrics import confusion_matrix
    except ModuleNotFoundError:
        logger.warning(
            "scikit-learn is not installed; skipping confusion matrix plot. "
            "Install it with: python -m pip install -r requirements.txt"
        )
        return

    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap="Blues", fmt="d")
    plt.title(f"Confusion Matrix {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    file_path = PLOTS_DIR / f"confusion_matrix_{model_name}.png"
    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(file_path, dpi=150)
        logger.info(f"Saved plot to {file_path}")
    except IOError as exc:
        logger.error("Failed to save confusion matrix plot")
        raise IOError("Failed to save confusion matrix plot") from exc
    finally:
        plt.close()


def plot_model_comparison(results_df: pd.DataFrame) -> None:
    """
    Plot model comparison results.

    Args:
        results_df (pd.DataFrame): DataFrame with model comparison metrics.

    Returns:
        None: This function returns None.
    """
    try:
        plt, sns = _import_plot_libs()
    except ModuleNotFoundError:
        return

    if results_df.empty or "model_name" not in results_df.columns:
        logger.warning("No results for model comparison plot")
        return

    metric = "accuracy" if "accuracy" in results_df.columns else "r2"
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=results_df[metric],
        y=results_df["model_name"],
        palette="Blues",
        orient="h",
    )
    plt.title("Model Comparison")
    plt.xlabel(metric)
    plt.ylabel("Model")
    plt.legend([metric])

    file_path = PLOTS_DIR / "model_comparison.png"
    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(file_path, dpi=150)
        logger.info(f"Saved plot to {file_path}")
    except IOError as exc:
        logger.error("Failed to save model comparison plot")
        raise IOError("Failed to save model comparison plot") from exc
    finally:
        plt.close()


def plot_training_history(history: Any, model_name: str) -> None:
    """
    Plot training history for a neural network model.

    Args:
        history (Any): Keras History object.
        model_name (str): Model name for labeling.

    Returns:
        None: This function returns None.
    """
    try:
        plt, _sns = _import_plot_libs()
    except ModuleNotFoundError:
        return

    history_dict = history.history if hasattr(history, "history") else {}
    if not history_dict:
        logger.warning("Empty history for training plot")
        return

    plt.figure(figsize=(12, 6))
    if "loss" in history_dict:
        plt.plot(history_dict["loss"], label="loss")
    if "val_loss" in history_dict:
        plt.plot(history_dict["val_loss"], label="val_loss")
    if "accuracy" in history_dict:
        plt.plot(history_dict["accuracy"], label="accuracy")
    if "val_accuracy" in history_dict:
        plt.plot(history_dict["val_accuracy"], label="val_accuracy")

    plt.title(f"Training History {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()

    file_path = PLOTS_DIR / f"training_history_{model_name}.png"
    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(file_path, dpi=150)
        logger.info(f"Saved plot to {file_path}")
    except IOError as exc:
        logger.error("Failed to save training history plot")
        raise IOError("Failed to save training history plot") from exc
    finally:
        plt.close()


def plot_feature_importance(model: Any, feature_names: List[str], model_name: str) -> None:
    """
    Plot feature importance for tree-based models.

    Args:
        model (Any): Trained model with feature_importances_.
        feature_names (List[str]): Feature names.
        model_name (str): Model name for labeling.

    Returns:
        None: This function returns None.
    """
    try:
        plt, sns = _import_plot_libs()
    except ModuleNotFoundError:
        return

    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not have feature_importances_")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=importances[indices], y=sorted_names, palette="viridis", orient="h")
    plt.title(f"Feature Importance {model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.legend(["importance"])

    file_path = PLOTS_DIR / f"feature_importance_{model_name}.png"
    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(file_path, dpi=150)
        logger.info(f"Saved plot to {file_path}")
    except IOError as exc:
        logger.error("Failed to save feature importance plot")
        raise IOError("Failed to save feature importance plot") from exc
    finally:
        plt.close()


def plot_pca_variance(pca_object: Any, model_name: str) -> None:
    """
    Plot PCA explained variance ratio.

    Args:
        pca_object (Any): Fitted PCA object.
        model_name (str): Name to include in plot title.

    Returns:
        None: This function returns None.
    """
    try:
        plt, _sns = _import_plot_libs()
    except ModuleNotFoundError:
        return

    if not hasattr(pca_object, "explained_variance_ratio_"):
        logger.warning("PCA object missing explained_variance_ratio_")
        return

    ratios = pca_object.explained_variance_ratio_
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(ratios) + 1), ratios, marker="o", label="variance")
    plt.title(f"PCA Variance {model_name}")
    plt.xlabel("Component")
    plt.ylabel("Explained Variance Ratio")
    plt.legend()

    file_path = PLOTS_DIR / "pca_variance.png"
    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(file_path, dpi=150)
        logger.info(f"Saved plot to {file_path}")
    except IOError as exc:
        logger.error("Failed to save PCA variance plot")
        raise IOError("Failed to save PCA variance plot") from exc
    finally:
        plt.close()
