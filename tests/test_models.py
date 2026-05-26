"""Tests for models and evaluators."""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.cleaner import clean_dataset
from src.data.loader import load_excel
from src.data.transformer import (
    apply_label_encoding,
    create_features,
    prepare_for_modeling,
)
from src.models.comparator import compare_all_models, save_comparison_report
from src.models.deep_learning import (
    build_deep_nn,
    build_shallow_nn,
    train_neural_network,
)
from src.models.evaluator import evaluate_classification, evaluate_regression
from src.models.ml_models import MLModelTrainer
from src.utils.config import (
    MODELS_DIR,
    REPORTS_DIR,
    TARGET_CLASSIFICATION_BINARY,
    TARGET_REGRESSION,
)
from src.visualization.plots import plot_correlation_heatmap, plot_data_distribution


DATA_PATH = "data/restaurante_foodvision.xlsx"


def _tensorflow_available() -> bool:
    """
    Check if TensorFlow is available.

    Args:
        None: This function does not take arguments.

    Returns:
        bool: True if TensorFlow can be imported.
    """
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


def _encode_all_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode all categorical and boolean columns.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Encoded dataset.
    """
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if not cat_cols:
        return df
    df_encoded, _ = apply_label_encoding(df, cat_cols)
    return df_encoded


def _prepare_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare a dataset split for modeling.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Train and test splits.
    """
    df = load_excel(DATA_PATH)
    df_clean, _ = clean_dataset(df)
    df_features = create_features(df_clean)
    df_encoded = _encode_all_categoricals(df_features)
    return prepare_for_modeling(df_encoded, TARGET_CLASSIFICATION_BINARY)


def _prepare_regression_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare a dataset split for regression modeling.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Train and test splits.
    """
    df = load_excel(DATA_PATH)
    df_clean, _ = clean_dataset(df)
    df_features = create_features(df_clean)
    df_encoded = _encode_all_categoricals(df_features)
    return prepare_for_modeling(df_encoded, TARGET_REGRESSION)


X_TRAIN_CLS, X_TEST_CLS, Y_TRAIN_CLS, Y_TEST_CLS = _prepare_splits()
X_TRAIN_REG, X_TEST_REG, Y_TRAIN_REG, Y_TEST_REG = _prepare_regression_splits()


def test_decision_tree_classification() -> Tuple[bool, str]:
    """
    Validate Decision Tree classification training.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    trainer = MLModelTrainer()
    try:
        model = trainer.train_decision_tree(X_TRAIN_CLS, Y_TRAIN_CLS, "classification")
        preds = trainer.predict(model, X_TEST_CLS)
    except Exception as exc:
        return False, f"Training failed: {exc}"
    if model is None:
        return False, "Model is None"
    if len(preds) != len(X_TEST_CLS):
        return False, "Prediction length mismatch"
    if not set(np.unique(preds)).issubset({0, 1}):
        return False, f"Unexpected predictions: {np.unique(preds)}"
    return True, "Decision tree classification passed"


def test_random_forest_classification() -> Tuple[bool, str]:
    """
    Validate Random Forest classification training.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    trainer = MLModelTrainer()
    try:
        model = trainer.train_random_forest(X_TRAIN_CLS, Y_TRAIN_CLS, "classification")
        preds = trainer.predict(model, X_TEST_CLS)
    except Exception as exc:
        return False, f"Training failed: {exc}"
    if model is None:
        return False, "Model is None"
    if len(preds) != len(X_TEST_CLS):
        return False, "Prediction length mismatch"
    if not set(np.unique(preds)).issubset({0, 1}):
        return False, f"Unexpected predictions: {np.unique(preds)}"
    return True, "Random forest classification passed"


def test_logistic_regression() -> Tuple[bool, str]:
    """
    Validate Logistic Regression training.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    trainer = MLModelTrainer()
    try:
        model = trainer.train_logistic_regression(X_TRAIN_CLS, Y_TRAIN_CLS)
        preds = trainer.predict(model, X_TEST_CLS)
    except Exception as exc:
        return False, f"Training failed: {exc}"
    if model is None:
        return False, "Model is None"
    if len(preds) != len(X_TEST_CLS):
        return False, "Prediction length mismatch"
    if not set(np.unique(preds)).issubset({0, 1}):
        return False, f"Unexpected predictions: {np.unique(preds)}"
    return True, "Logistic regression passed"


def test_knn_classification() -> Tuple[bool, str]:
    """
    Validate KNN classification training.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    trainer = MLModelTrainer()
    try:
        model = trainer.train_knn(X_TRAIN_CLS, Y_TRAIN_CLS, "classification")
        preds = trainer.predict(model, X_TEST_CLS)
    except Exception as exc:
        return False, f"Training failed: {exc}"
    if model is None:
        return False, "Model is None"
    if len(preds) != len(X_TEST_CLS):
        return False, "Prediction length mismatch"
    if not set(np.unique(preds)).issubset({0, 1}):
        return False, f"Unexpected predictions: {np.unique(preds)}"
    return True, "KNN classification passed"


def test_decision_tree_regression() -> Tuple[bool, str]:
    """
    Validate Decision Tree regression training.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    trainer = MLModelTrainer()
    try:
        model = trainer.train_decision_tree(X_TRAIN_REG, Y_TRAIN_REG, "regression")
        preds = trainer.predict(model, X_TEST_REG)
    except Exception as exc:
        return False, f"Training failed: {exc}"
    if model is None:
        return False, "Model is None"
    if len(preds) != len(X_TEST_REG):
        return False, "Prediction length mismatch"
    if not np.issubdtype(np.array(preds).dtype, np.number):
        return False, f"Non-numeric predictions: {np.array(preds).dtype}"
    return True, "Decision tree regression passed"


def test_random_forest_regression() -> Tuple[bool, str]:
    """
    Validate Random Forest regression training.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    trainer = MLModelTrainer()
    try:
        model = trainer.train_random_forest(X_TRAIN_REG, Y_TRAIN_REG, "regression")
        preds = trainer.predict(model, X_TEST_REG)
    except Exception as exc:
        return False, f"Training failed: {exc}"
    if model is None:
        return False, "Model is None"
    if len(preds) != len(X_TEST_REG):
        return False, "Prediction length mismatch"
    if not np.issubdtype(np.array(preds).dtype, np.number):
        return False, f"Non-numeric predictions: {np.array(preds).dtype}"
    return True, "Random forest regression passed"


def test_knn_regression() -> Tuple[bool, str]:
    """
    Validate KNN regression training.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    trainer = MLModelTrainer()
    try:
        model = trainer.train_knn(X_TRAIN_REG, Y_TRAIN_REG, "regression")
        preds = trainer.predict(model, X_TEST_REG)
    except Exception as exc:
        return False, f"Training failed: {exc}"
    if model is None:
        return False, "Model is None"
    if len(preds) != len(X_TEST_REG):
        return False, "Prediction length mismatch"
    if not np.issubdtype(np.array(preds).dtype, np.number):
        return False, f"Non-numeric predictions: {np.array(preds).dtype}"
    return True, "KNN regression passed"


def test_evaluate_classification_metrics() -> Tuple[bool, str]:
    """
    Validate classification evaluator metrics.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    trainer = MLModelTrainer()
    try:
        model = trainer.train_decision_tree(X_TRAIN_CLS, Y_TRAIN_CLS, "classification")
        preds = trainer.predict(model, X_TEST_CLS)
        results = evaluate_classification(Y_TEST_CLS, preds, "decision_tree")
    except Exception as exc:
        return False, f"Evaluation failed: {exc}"
    required = {"model_name", "accuracy", "precision", "recall", "f1_score", "task"}
    missing = required - set(results.keys())
    if missing:
        return False, f"Missing keys: {sorted(missing)}"
    for key in ["accuracy", "precision", "recall", "f1_score"]:
        value = results.get(key, -1.0)
        if not (0.0 <= value <= 1.0):
            return False, f"Metric {key} out of range: {value}"
    return True, "Classification metrics valid"


def test_evaluate_regression_metrics() -> Tuple[bool, str]:
    """
    Validate regression evaluator metrics.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    trainer = MLModelTrainer()
    try:
        model = trainer.train_decision_tree(X_TRAIN_REG, Y_TRAIN_REG, "regression")
        preds = trainer.predict(model, X_TEST_REG)
        results = evaluate_regression(Y_TEST_REG, preds, "decision_tree")
    except Exception as exc:
        return False, f"Evaluation failed: {exc}"
    required = {"model_name", "mae", "mse", "rmse", "r2", "task"}
    missing = required - set(results.keys())
    if missing:
        return False, f"Missing keys: {sorted(missing)}"
    for key in ["mae", "mse", "rmse"]:
        value = results.get(key, -1.0)
        if value < 0:
            return False, f"Metric {key} negative: {value}"
    if results.get("r2", 2.0) > 1.0:
        return False, f"Metric r2 out of range: {results.get('r2')}"
    return True, "Regression metrics valid"


def test_compare_all_models() -> Tuple[bool, str]:
    """
    Validate model comparison sorting.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    results_list: List[Dict[str, float]] = [
        {"model_name": "a", "accuracy": 0.8, "task": "classification"},
        {"model_name": "b", "accuracy": 0.9, "task": "classification"},
        {"model_name": "c", "accuracy": 0.7, "task": "classification"},
        {"model_name": "d", "accuracy": 0.85, "task": "classification"},
    ]
    df = compare_all_models(results_list)
    if not isinstance(df, pd.DataFrame):
        return False, "compare_all_models did not return DataFrame"
    if "model_name" not in df.columns:
        return False, f"Missing model_name column, got {df.columns.tolist()}"
    accuracies = df["accuracy"].tolist()
    if accuracies != sorted(accuracies, reverse=True):
        return False, f"Results not sorted by accuracy: {accuracies}"
    return True, "Model comparison sorting valid"


def test_save_comparison_report() -> Tuple[bool, str]:
    """
    Validate saving model comparison report.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = REPORTS_DIR / "test_report.csv"
    df = pd.DataFrame(
        [
            {"model_name": "a", "accuracy": 0.8, "task": "classification"},
            {"model_name": "b", "accuracy": 0.9, "task": "classification"},
        ]
    )
    save_comparison_report(df, file_path)
    if not file_path.exists():
        return False, "Report file not created"
    try:
        _ = pd.read_csv(file_path)
    except OSError as exc:
        return False, f"Failed to read report: {exc}"
    return True, "Comparison report saved and readable"


def test_models_saved_to_disk() -> Tuple[bool, str]:
    """
    Validate model persistence on disk.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    trainer = MLModelTrainer()
    try:
        _ = trainer.train_decision_tree(X_TRAIN_CLS, Y_TRAIN_CLS, "classification")
        _ = trainer.train_random_forest(X_TRAIN_CLS, Y_TRAIN_CLS, "classification")
        _ = trainer.train_logistic_regression(X_TRAIN_CLS, Y_TRAIN_CLS)
        _ = trainer.train_knn(X_TRAIN_CLS, Y_TRAIN_CLS, "classification")
    except Exception as exc:
        return False, f"Training failed: {exc}"

    model_files = list(MODELS_DIR.glob("*.pkl"))
    if len(model_files) >= 4:
        return True, f"Found {len(model_files)} model files"
    return False, f"Only found {len(model_files)} model files"


def test_shallow_nn_builds() -> Tuple[bool, str]:
    """
    Validate shallow neural network build.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    if not _tensorflow_available():
        return True, "TensorFlow not available, test skipped"
    try:
        model = build_shallow_nn(input_dim=10, output_units=2, task="classification")
    except Exception as exc:
        return False, f"Build failed: {exc}"
    if model.__class__.__name__ != "Sequential":
        return False, "Model is not Sequential"
    dense_layers = [layer for layer in model.layers if layer.__class__.__name__ == "Dense"]
    if len(dense_layers) != 2:
        return False, f"Expected 2 Dense layers, found {len(dense_layers)}"
    try:
        _ = model.predict(np.random.rand(5, 10), verbose=0)
    except Exception as exc:
        return False, f"Predict failed: {exc}"
    return True, "Shallow neural network builds"


def test_deep_nn_builds() -> Tuple[bool, str]:
    """
    Validate deep neural network build.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    if not _tensorflow_available():
        return True, "TensorFlow not available, test skipped"
    try:
        model = build_deep_nn(input_dim=10, output_units=2, task="classification")
    except Exception as exc:
        return False, f"Build failed: {exc}"
    if model.__class__.__name__ != "Sequential":
        return False, "Model is not Sequential"
    if len(model.layers) < 5:
        return False, f"Expected at least 5 layers, found {len(model.layers)}"
    return True, "Deep neural network builds"


def test_nn_training_records_time() -> Tuple[bool, str]:
    """
    Validate neural network training time and history.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    if not _tensorflow_available():
        return True, "TensorFlow not available, test skipped"
    try:
        model = build_shallow_nn(input_dim=10, output_units=2, task="classification")
        X = np.random.rand(200, 10)
        y = np.random.randint(0, 2, 200)
        start = time.time()
        history = train_neural_network(model, X, y, epochs=3, batch_size=32)
        elapsed = time.time() - start
    except Exception as exc:
        return False, f"Training failed: {exc}"
    if elapsed >= 60:
        return False, f"Training took too long: {elapsed:.2f} seconds"
    if "loss" not in history.history or "val_loss" not in history.history:
        return False, "History missing loss or val_loss"
    return True, "Neural network training valid"


def test_plots_saved_to_disk() -> Tuple[bool, str]:
    """
    Validate plot generation and saving.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    df = load_excel(DATA_PATH)
    df_clean, _ = clean_dataset(df)
    plot_data_distribution(df_clean)
    plot_correlation_heatmap(df_clean)
    plot_files = list((MODELS_DIR.parent / "plots").glob("*.png"))
    if len(plot_files) >= 2:
        return True, f"Found {len(plot_files)} plot files"
    return False, f"Only found {len(plot_files)} plot files"
