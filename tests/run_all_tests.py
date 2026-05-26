"""Master test runner for FoodVision AI validations."""

from __future__ import annotations

from typing import Callable, List, Tuple

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.test_cleaner import (
    test_cleaning_report_structure,
    test_fix_negative_values,
    test_fix_outliers,
    test_impute_missing_values,
    test_remove_duplicates,
    test_standardize_city_names,
    test_validate_ranges,
)
from tests.test_models import (
    test_compare_all_models,
    test_decision_tree_classification,
    test_decision_tree_regression,
    test_deep_nn_builds,
    test_evaluate_classification_metrics,
    test_evaluate_regression_metrics,
    test_knn_classification,
    test_knn_regression,
    test_logistic_regression,
    test_models_saved_to_disk,
    test_nn_training_records_time,
    test_plots_saved_to_disk,
    test_random_forest_classification,
    test_random_forest_regression,
    test_save_comparison_report,
    test_shallow_nn_builds,
)
from tests.test_transformer import (
    test_feature_engineering_categoria_consumo,
    test_feature_engineering_cliente_premium,
    test_feature_engineering_nivel_demora,
    test_feature_engineering_promedio_compra,
    test_label_encoding,
    test_minmax_scaling,
    test_one_hot_encoding,
    test_pca,
    test_prepare_for_modeling,
    test_zscore_scaling,
)


TestFunc = Callable[[], Tuple[bool, str]]


def _run_test(name: str, func: TestFunc) -> Tuple[bool, str]:
    """
    Run a single test and handle exceptions.

    Args:
        name (str): Test name.
        func (TestFunc): Test function.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    try:
        passed, reason = func()
        return passed, reason
    except Exception as exc:
        return False, f"{name} raised exception: {exc}"


def main() -> None:
    """
    Run all tests and print a summary report.

    Args:
        None: This function does not take arguments.

    Returns:
        None: This function returns None.
    """
    tests: List[Tuple[str, str, TestFunc]] = [
        ("Part 3", "test_standardize_city_names", test_standardize_city_names),
        ("Part 3", "test_fix_negative_values", test_fix_negative_values),
        ("Part 3", "test_fix_outliers", test_fix_outliers),
        ("Part 3", "test_impute_missing_values", test_impute_missing_values),
        ("Part 3", "test_remove_duplicates", test_remove_duplicates),
        ("Part 3", "test_validate_ranges", test_validate_ranges),
        ("Part 3", "test_cleaning_report_structure", test_cleaning_report_structure),
        ("Part 4", "test_minmax_scaling", test_minmax_scaling),
        ("Part 4", "test_zscore_scaling", test_zscore_scaling),
        ("Part 4", "test_one_hot_encoding", test_one_hot_encoding),
        ("Part 4", "test_label_encoding", test_label_encoding),
        ("Part 4", "test_feature_engineering_categoria_consumo", test_feature_engineering_categoria_consumo),
        ("Part 4", "test_feature_engineering_nivel_demora", test_feature_engineering_nivel_demora),
        ("Part 4", "test_feature_engineering_cliente_premium", test_feature_engineering_cliente_premium),
        ("Part 4", "test_feature_engineering_promedio_compra", test_feature_engineering_promedio_compra),
        ("Part 4", "test_pca", test_pca),
        ("Part 4", "test_prepare_for_modeling", test_prepare_for_modeling),
        ("Part 5", "test_decision_tree_classification", test_decision_tree_classification),
        ("Part 5", "test_random_forest_classification", test_random_forest_classification),
        ("Part 5", "test_logistic_regression", test_logistic_regression),
        ("Part 5", "test_knn_classification", test_knn_classification),
        ("Part 5", "test_decision_tree_regression", test_decision_tree_regression),
        ("Part 5", "test_random_forest_regression", test_random_forest_regression),
        ("Part 5", "test_knn_regression", test_knn_regression),
        ("Part 5", "test_evaluate_classification_metrics", test_evaluate_classification_metrics),
        ("Part 5", "test_evaluate_regression_metrics", test_evaluate_regression_metrics),
        ("Part 5", "test_compare_all_models", test_compare_all_models),
        ("Part 5", "test_save_comparison_report", test_save_comparison_report),
        ("Part 5", "test_models_saved_to_disk", test_models_saved_to_disk),
        ("Part 5", "test_shallow_nn_builds", test_shallow_nn_builds),
        ("Part 5", "test_deep_nn_builds", test_deep_nn_builds),
        ("Part 5", "test_nn_training_records_time", test_nn_training_records_time),
        ("Part 5", "test_plots_saved_to_disk", test_plots_saved_to_disk),
    ]

    print("=" * 59)
    print("FoodVision AI - Validation Report")
    print("=" * 59)

    passed_count = 0
    failed_count = 0

    for part, name, func in tests:
        passed, reason = _run_test(name, func)
        status = "PASS" if passed else "FAIL"
        if passed:
            passed_count += 1
        else:
            failed_count += 1
        print(f"{status}  [{part}] {name}  -- {reason}")

    total = len(tests)
    print("=" * 59)
    print(f"Total: {total} tests | Passed: {passed_count} | Failed: {failed_count}")
    print("=" * 59)


if __name__ == "__main__":
    main()
