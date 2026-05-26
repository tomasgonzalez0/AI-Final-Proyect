"""FoodVision AI main application entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd

from src.data.loader import display_dataset_info, load_excel
from src.data.cleaner import clean_dataset
from src.data.transformer import (
    apply_label_encoding,
    apply_minmax_scaling,
    apply_pca,
    create_features,
    prepare_for_modeling,
)
from src.models.ml_models import MLModelTrainer
from src.models.deep_learning import (
    build_deep_nn,
    build_shallow_nn,
    predict_nn,
    save_model,
    train_neural_network,
)
from src.models.evaluator import (
    evaluate_classification,
    evaluate_regression,
    print_metrics_table,
)
from src.models.comparator import compare_all_models, save_comparison_report
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_data_distribution,
    plot_feature_importance,
    plot_model_comparison,
    plot_pca_variance,
    plot_training_history,
)
from src.utils.config import (
    CATEGORICAL_COLUMNS,
    DATA_DIR,
    EXCEL_FILENAME,
    NUMERIC_COLUMNS,
    REPORTS_DIR,
    TARGET_CLASSIFICATION_BINARY,
    TARGET_REGRESSION,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)

app_state: Dict[str, Any] = {
    "df_raw": None,
    "df_clean": None,
    "df_transformed": None,
    "results": [],
    "best_model": None,
}


def display_menu() -> None:
    """
    Display the main menu.

    Args:
        None: This function does not take arguments.

    Returns:
        None: This function returns None.
    """
    print("=== FoodVision AI - Restaurant Intelligence System ===")
    print("1. Load and display dataset")
    print("2. Clean data automatically")
    print("3. Transform variables")
    print("4. Train ML models - classification")
    print("5. Train ML models - regression")
    print("6. Train Neural Networks")
    print("7. Compare all models")
    print("8. Generate visualizations")
    print("9. Make a prediction - interactive")
    print("0. Exit")


def _check_state(key: str, step_number: int) -> bool:
    """
    Validate that a required state exists.

    Args:
        key (str): State key to check.
        step_number (int): Step number required to proceed.

    Returns:
        bool: True if the state exists, False otherwise.
    """
    if app_state.get(key) is None:
        print(f"Please complete step {step_number} first")
        return False
    return True


def _select_best_model(results: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """
    Select the best model from results.

    Args:
        results (List[Dict[str, Any]]): List of evaluation results.

    Returns:
        Dict[str, Any] | None: Best model metadata or None.
    """
    if not results:
        return None

    classification = [r for r in results if r.get("task") == "classification"]
    regression = [r for r in results if r.get("task") == "regression"]

    if classification:
        return max(classification, key=lambda r: r.get("accuracy", 0.0))
    if regression:
        return max(regression, key=lambda r: r.get("r2", 0.0))
    return None


def _prepare_transformed_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create features, encode categoricals, and scale numeric columns.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Transformed dataset and artifacts.
    """
    df_features = create_features(df)
    engineered_cols = ["categoria_consumo", "nivel_demora", "cliente_premium"]
    categorical_cols = [
        col for col in CATEGORICAL_COLUMNS + engineered_cols if col in df_features.columns
    ]

    df_encoded, encoders = apply_label_encoding(df_features, categorical_cols)

    numeric_cols = [
        col for col in NUMERIC_COLUMNS + ["promedio_compra_cliente"] if col in df_encoded.columns
    ]
    df_scaled, scaler = apply_minmax_scaling(df_encoded, numeric_cols)

    artifacts = {"encoders": encoders, "scaler": scaler, "numeric_cols": numeric_cols}
    return df_scaled, artifacts


def _train_and_evaluate_classification(df: pd.DataFrame) -> None:
    """
    Train and evaluate traditional classifiers.

    Args:
        df (pd.DataFrame): Transformed dataset.

    Returns:
        None: This function returns None.
    """
    trainer = MLModelTrainer()
    X_train, X_test, y_train, y_test = prepare_for_modeling(
        df,
        TARGET_CLASSIFICATION_BINARY,
    )

    models = [
        ("decision_tree", lambda: trainer.train_decision_tree(X_train, y_train, "classification")),
        ("random_forest", lambda: trainer.train_random_forest(X_train, y_train, "classification")),
        ("logistic_regression", lambda: trainer.train_logistic_regression(X_train, y_train)),
        ("knn", lambda: trainer.train_knn(X_train, y_train, "classification")),
    ]

    for name, train_fn in models:
        try:
            model = train_fn()
            preds = trainer.predict(model, X_test)
            results = evaluate_classification(y_test, preds, name)
            results["model"] = model
            app_state["results"].append(results)
        except ValueError as exc:
            logger.error(f"Training failed for {name}: {exc}")


def _train_and_evaluate_regression(df: pd.DataFrame) -> None:
    """
    Train and evaluate traditional regressors.

    Args:
        df (pd.DataFrame): Transformed dataset.

    Returns:
        None: This function returns None.
    """
    trainer = MLModelTrainer()
    X_train, X_test, y_train, y_test = prepare_for_modeling(df, TARGET_REGRESSION)

    models = [
        ("decision_tree", lambda: trainer.train_decision_tree(X_train, y_train, "regression")),
        ("random_forest", lambda: trainer.train_random_forest(X_train, y_train, "regression")),
        ("linear_regression", lambda: trainer.train_logistic_regression(X_train, y_train)),
        ("knn", lambda: trainer.train_knn(X_train, y_train, "regression")),
    ]

    for name, train_fn in models:
        try:
            model = train_fn()
            preds = trainer.predict(model, X_test)
            results = evaluate_regression(y_test, preds, name)
            results["model"] = model
            app_state["results"].append(results)
        except ValueError as exc:
            logger.error(f"Training failed for {name}: {exc}")


def _train_and_evaluate_neural_networks(df: pd.DataFrame) -> None:
    """
    Train and evaluate shallow and deep neural networks.

    Args:
        df (pd.DataFrame): Transformed dataset.

    Returns:
        None: This function returns None.
    """
    X_train, X_test, y_train, y_test = prepare_for_modeling(
        df,
        TARGET_CLASSIFICATION_BINARY,
    )

    num_classes = int(pd.Series(y_train).nunique())
    task = "binary" if num_classes == 2 else "classification"
    output_units = 1 if task == "binary" else num_classes

    models = [
        ("shallow_nn", build_shallow_nn),
        ("deep_nn", build_deep_nn),
    ]

    for name, builder in models:
        try:
            model = builder(X_train.shape[1], output_units, task)
            history = train_neural_network(model, X_train, y_train)
            save_model(model, name)
            preds = predict_nn(model, X_test)
            if task == "binary":
                pred_labels = (preds.reshape(-1) >= 0.5).astype(int)
            else:
                pred_labels = np.argmax(preds, axis=1)
            results = evaluate_classification(y_test, pred_labels, name)
            results["model"] = model
            results["history"] = history
            app_state["results"].append(results)
        except (ValueError, RuntimeError) as exc:
            logger.error(f"Neural network training failed for {name}: {exc}")


def _generate_visualizations(df: pd.DataFrame) -> None:
    """
    Generate plots for the current dataset and models.

    Args:
        df (pd.DataFrame): Transformed dataset.

    Returns:
        None: This function returns None.
    """
    plot_data_distribution(df)
    plot_correlation_heatmap(df)

    trainer = MLModelTrainer()
    X_train, X_test, y_train, y_test = prepare_for_modeling(
        df,
        TARGET_CLASSIFICATION_BINARY,
    )

    model_dt = trainer.train_decision_tree(X_train, y_train, "classification")
    preds_dt = trainer.predict(model_dt, X_test)
    plot_confusion_matrix(y_test, preds_dt, "decision_tree")

    model_rf = trainer.train_random_forest(X_train, y_train, "classification")
    preds_rf = trainer.predict(model_rf, X_test)
    plot_confusion_matrix(y_test, preds_rf, "random_forest")

    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else []
    plot_feature_importance(model_rf, feature_names, "random_forest")

    results_df = compare_all_models(app_state["results"])
    if not results_df.empty:
        plot_model_comparison(results_df)

    num_df = df.select_dtypes(include=["number"])
    if not num_df.empty:
        df_pca, pca_object = apply_pca(num_df, num_df.shape[1])
        _ = df_pca
        plot_pca_variance(pca_object, "pca")

    histories = [r for r in app_state["results"] if "history" in r]
    for item in histories:
        plot_training_history(item["history"], item["model_name"])


def _interactive_prediction(df: pd.DataFrame) -> None:
    """
    Run an interactive prediction using the best available model.

    Args:
        df (pd.DataFrame): Clean dataset for preprocessing context.

    Returns:
        None: This function returns None.
    """
    if app_state.get("best_model") is None:
        print("Please complete step 4 first")
        return

    best_model_info = app_state["best_model"]
    model = best_model_info.get("model")
    task = best_model_info.get("task")

    input_data = {
        "ciudad": input("ciudad: "),
        "tipo_comida": input("tipo_comida: "),
        "hora_pedido": int(input("hora_pedido: ")),
        "cantidad_productos": int(input("cantidad_productos: ")),
        "valor_total": float(input("valor_total: ")),
        "tiempo_preparacion": float(input("tiempo_preparacion: ")),
        "metodo_pago": input("metodo_pago: "),
        "clima": input("clima: "),
    }

    df_input = pd.DataFrame([input_data])
    df_full = pd.concat([df, df_input], ignore_index=True)
    df_transformed, _ = _prepare_transformed_data(df_full)
    X_input = df_transformed.tail(1)

    if task == "regression":
        prediction = model.predict(X_input)[0]
        print(f"Prediction: {prediction}")
        return

    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)
        pred_label = int(np.argmax(probs, axis=1)[0])
        confidence = float(np.max(probs))
    else:
        preds = model.predict(X_input)
        pred_label = int(preds[0])
        confidence = 1.0

    print(f"Prediction: {pred_label}")
    print(f"Confidence: {confidence}")


def main() -> None:
    """
    Run the FoodVision AI application.

    Args:
        None: This function does not take arguments.

    Returns:
        None: This function returns None.
    """
    while True:
        display_menu()
        option = input("Select an option: ").strip()

        if option == "1":
            try:
                logger.info("Option 1 selected")
                file_path = DATA_DIR / EXCEL_FILENAME
                df = load_excel(file_path)
                display_dataset_info(df)
                app_state["df_raw"] = df
            except (FileNotFoundError, ValueError, OSError) as exc:
                logger.error(f"Failed to load dataset: {exc}")
                print(f"Error: {exc}")

        elif option == "2":
            if not _check_state("df_raw", 1):
                continue
            try:
                logger.info("Option 2 selected")
                df_clean, report = clean_dataset(app_state["df_raw"])
                app_state["df_clean"] = df_clean
                print(report)
            except ValueError as exc:
                logger.error(f"Data cleaning failed: {exc}")
                print(f"Error: {exc}")

        elif option == "3":
            if not _check_state("df_clean", 2):
                continue
            try:
                logger.info("Option 3 selected")
                df_transformed, _ = _prepare_transformed_data(app_state["df_clean"])
                app_state["df_transformed"] = df_transformed
                logger.info("Data transformation completed")
            except ValueError as exc:
                logger.error(f"Data transformation failed: {exc}")
                print(f"Error: {exc}")

        elif option == "4":
            if not _check_state("df_transformed", 3):
                continue
            try:
                logger.info("Option 4 selected")
                _train_and_evaluate_classification(app_state["df_transformed"])
                best = _select_best_model(app_state["results"])
                if best is not None:
                    app_state["best_model"] = best
            except ValueError as exc:
                logger.error(f"Classification training failed: {exc}")
                print(f"Error: {exc}")

        elif option == "5":
            if not _check_state("df_transformed", 3):
                continue
            try:
                logger.info("Option 5 selected")
                _train_and_evaluate_regression(app_state["df_transformed"])
                best = _select_best_model(app_state["results"])
                if best is not None:
                    app_state["best_model"] = best
            except ValueError as exc:
                logger.error(f"Regression training failed: {exc}")
                print(f"Error: {exc}")

        elif option == "6":
            if not _check_state("df_transformed", 3):
                continue
            try:
                logger.info("Option 6 selected")
                _train_and_evaluate_neural_networks(app_state["df_transformed"])
                best = _select_best_model(app_state["results"])
                if best is not None:
                    app_state["best_model"] = best
            except (ValueError, RuntimeError) as exc:
                logger.error(f"Neural network training failed: {exc}")
                print(f"Error: {exc}")

        elif option == "7":
            if not app_state["results"]:
                print("Please complete step 4 first")
                continue
            try:
                logger.info("Option 7 selected")
                results_df = compare_all_models(app_state["results"])
                REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                report_path = REPORTS_DIR / "model_comparison.csv"
                save_comparison_report(results_df, report_path)
                print_metrics_table(app_state["results"])
            except (ValueError, OSError) as exc:
                logger.error(f"Model comparison failed: {exc}")
                print(f"Error: {exc}")

        elif option == "8":
            if not _check_state("df_transformed", 3):
                continue
            try:
                logger.info("Option 8 selected")
                _generate_visualizations(app_state["df_transformed"])
            except (ValueError, OSError, IOError) as exc:
                logger.error(f"Visualization failed: {exc}")
                print(f"Error: {exc}")

        elif option == "9":
            if not _check_state("df_clean", 2):
                continue
            try:
                logger.info("Option 9 selected")
                _interactive_prediction(app_state["df_clean"])
            except (ValueError, OSError) as exc:
                logger.error(f"Prediction failed: {exc}")
                print(f"Error: {exc}")

        elif option == "0":
            print("Exiting FoodVision AI")
            break

        else:
            print("Invalid option, please try again")


if __name__ == "__main__":
    main()
