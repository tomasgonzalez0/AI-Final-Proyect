"""Tests for data transformation functions."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from src.data.cleaner import clean_dataset
from src.data.loader import load_excel
from src.data.transformer import (
    apply_label_encoding,
    apply_minmax_scaling,
    apply_one_hot_encoding,
    apply_pca,
    apply_zscore_scaling,
    create_features,
    prepare_for_modeling,
)
from src.utils.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_CLASSIFICATION_BINARY


DATA_PATH = "data/restaurante_foodvision.xlsx"


def _load_clean_data() -> pd.DataFrame:
    """
    Load and clean the dataset.

    Args:
        None: This function does not take arguments.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    df = load_excel(DATA_PATH)
    df_clean, _ = clean_dataset(df)
    return df_clean


def test_minmax_scaling() -> Tuple[bool, str]:
    """
    Validate MinMax scaling ranges.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    df_clean = _load_clean_data()
    df_scaled, _ = apply_minmax_scaling(df_clean, NUMERIC_COLUMNS)
    tolerance = 1e-6

    for col in NUMERIC_COLUMNS:
        if col in df_scaled.columns:
            series = df_scaled[col]
            if ((series < -tolerance) | (series > 1 + tolerance)).any():
                bad_vals = series[(series < -tolerance) | (series > 1 + tolerance)].head(5)
                return False, f"Column {col} out of range values: {bad_vals.tolist()}"
    return True, "All MinMax scaled values within range"


def test_zscore_scaling() -> Tuple[bool, str]:
    """
    Validate Z-score scaling mean and std.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    df_clean = _load_clean_data()
    df_scaled, _ = apply_zscore_scaling(df_clean, NUMERIC_COLUMNS)

    for col in NUMERIC_COLUMNS:
        if col in df_scaled.columns:
            mean_val = float(df_scaled[col].mean())
            std_val = float(df_scaled[col].std(ddof=0))
            if abs(mean_val) >= 0.01 or abs(std_val - 1.0) >= 0.1:
                return False, f"Column {col} mean {mean_val} std {std_val}"
    return True, "All Z-score scaled columns within mean and std thresholds"


def test_one_hot_encoding() -> Tuple[bool, str]:
    """
    Validate one-hot encoding results.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    df_clean = _load_clean_data()
    df_encoded = apply_one_hot_encoding(df_clean, ["metodo_pago", "clima"])
    remaining = [col for col in ["metodo_pago", "clima"] if col in df_encoded.columns]

    new_cols = [col for col in df_encoded.columns if col.startswith("metodo_pago_")]
    if remaining:
        return False, f"Original columns still exist: {remaining}"
    if not new_cols:
        return False, "No new one-hot columns created"
    return True, "One-hot encoding successful"


def test_label_encoding() -> Tuple[bool, str]:
    """
    Validate label encoding outputs.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    df_clean = _load_clean_data()
    df_encoded, encoders = apply_label_encoding(df_clean, ["ciudad", "tipo_comida"])

    for col in ["ciudad", "tipo_comida"]:
        if col in df_encoded.columns and not pd.api.types.is_integer_dtype(df_encoded[col]):
            return False, f"Column {col} is not integer dtype"

    missing = [key for key in ["ciudad", "tipo_comida"] if key not in encoders]
    if missing:
        return False, f"Missing encoders for: {missing}"
    return True, "Label encoding successful"


def test_feature_engineering_categoria_consumo() -> Tuple[bool, str]:
    """
    Validate categoria_consumo feature.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    df_clean = _load_clean_data()
    df_features = create_features(df_clean)
    if "categoria_consumo" not in df_features.columns:
        return False, "categoria_consumo column missing"

    values = set(df_features["categoria_consumo"].dropna().unique())
    expected = {"Bajo", "Medio", "Alto"}
    if values - expected:
        return False, f"Unexpected values: {sorted(values - expected)}"
    if df_features["categoria_consumo"].isnull().any():
        return False, "categoria_consumo contains nulls"
    return True, "categoria_consumo values valid"


def test_feature_engineering_nivel_demora() -> Tuple[bool, str]:
    """
    Validate nivel_demora feature.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    df_clean = _load_clean_data()
    df_features = create_features(df_clean)
    if "nivel_demora" not in df_features.columns:
        return False, "nivel_demora column missing"

    values = set(df_features["nivel_demora"].dropna().unique())
    expected = {"Normal", "Moderado", "Critico"}
    if values - expected:
        return False, f"Unexpected values: {sorted(values - expected)}"
    return True, "nivel_demora values valid"


def test_feature_engineering_cliente_premium() -> Tuple[bool, str]:
    """
    Validate cliente_premium feature.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    df_clean = _load_clean_data()
    df_features = create_features(df_clean)
    if "cliente_premium" not in df_features.columns:
        return False, "cliente_premium column missing"

    if df_features["cliente_premium"].dtype != bool:
        return False, "cliente_premium is not boolean"

    expected_true = df_features[
        (df_features["cliente_frecuente"] == "Si")
        & (df_features["calificacion_cliente"] >= 4)
    ]
    if not df_features.loc[expected_true.index, "cliente_premium"].any():
        return False, "cliente_premium has no expected True values"
    return True, "cliente_premium values valid"


def test_feature_engineering_promedio_compra() -> Tuple[bool, str]:
    """
    Validate promedio_compra_cliente feature.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    df_clean = _load_clean_data()
    df_features = create_features(df_clean)
    if "promedio_compra_cliente" not in df_features.columns:
        return False, "promedio_compra_cliente column missing"

    series = df_features["promedio_compra_cliente"]
    inf_count = int(np.isinf(series).sum())
    null_count = int(series.isnull().sum())
    if inf_count > 0 or null_count > 0:
        return False, f"Inf count {inf_count} null count {null_count}"
    return True, "promedio_compra_cliente values valid"


def test_pca() -> Tuple[bool, str]:
    """
    Validate PCA output shape and attributes.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    df_clean = _load_clean_data()
    df_features = create_features(df_clean)
    df_encoded, _ = apply_label_encoding(df_features, CATEGORICAL_COLUMNS)
    numeric_df = df_encoded.select_dtypes(include=["number"])
    df_pca, pca_obj = apply_pca(numeric_df, 3)

    if df_pca.shape[1] != 3:
        return False, f"PCA columns: {df_pca.shape[1]}"
    if not hasattr(pca_obj, "explained_variance_ratio_"):
        return False, "PCA object missing explained_variance_ratio_"
    ratio_len = len(pca_obj.explained_variance_ratio_)
    ratio_sum = float(np.sum(pca_obj.explained_variance_ratio_))
    if ratio_len != 3 or ratio_sum > 1.0:
        return False, f"PCA ratio len {ratio_len} sum {ratio_sum}"
    return True, "PCA output valid"


def test_prepare_for_modeling() -> Tuple[bool, str]:
    """
    Validate train test split and target encoding.

    Args:
        None: This function does not take arguments.

    Returns:
        Tuple[bool, str]: Test result and reason.
    """
    df_clean = _load_clean_data()
    df_features = create_features(df_clean)
    df_encoded, _ = apply_label_encoding(df_features, CATEGORICAL_COLUMNS)
    df_encoded, _ = apply_label_encoding(df_encoded, [TARGET_CLASSIFICATION_BINARY])
    X_train, X_test, y_train, y_test = prepare_for_modeling(
        df_encoded,
        TARGET_CLASSIFICATION_BINARY,
    )

    if X_train.empty or X_test.empty or len(y_train) == 0 or len(y_test) == 0:
        return False, "Train test splits are empty"

    if len(X_train) + len(X_test) != len(df_encoded):
        return False, "Row count mismatch in split"

    unique_targets = set(pd.Series(y_train).unique().tolist())
    if not unique_targets.issubset({0, 1}):
        return False, f"Unexpected target values: {sorted(unique_targets)}"
    return True, "prepare_for_modeling output valid"
