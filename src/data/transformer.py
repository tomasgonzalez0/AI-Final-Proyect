"""Data transformation utilities for FoodVision AI."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from src.utils.config import COLUMNS_TO_DROP, RANDOM_STATE, TEST_SIZE
from src.utils.logger import get_logger


logger = get_logger(__name__)


def apply_minmax_scaling(
    df: pd.DataFrame, columns: List[str]
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Apply MinMax scaling to selected columns.

    Args:
        df (pd.DataFrame): Input dataset.
        columns (List[str]): Columns to scale.

    Returns:
        Tuple[pd.DataFrame, MinMaxScaler]: Scaled dataset and fitted scaler.
    """
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    valid_columns = [col for col in columns if col in df.columns]

    if not valid_columns:
        logger.warning("No valid columns for MinMax scaling")
        return df_scaled, scaler

    try:
        df_scaled[valid_columns] = scaler.fit_transform(df_scaled[valid_columns])
        logger.info("Applied MinMax scaling")
        return df_scaled, scaler
    except ValueError as exc:
        logger.error("Failed MinMax scaling")
        raise ValueError("Failed MinMax scaling") from exc


def apply_zscore_scaling(
    df: pd.DataFrame, columns: List[str]
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Apply Z-score scaling to selected columns.

    Args:
        df (pd.DataFrame): Input dataset.
        columns (List[str]): Columns to scale.

    Returns:
        Tuple[pd.DataFrame, StandardScaler]: Scaled dataset and fitted scaler.
    """
    scaler = StandardScaler()
    df_scaled = df.copy()
    valid_columns = [col for col in columns if col in df.columns]

    if not valid_columns:
        logger.warning("No valid columns for Z-score scaling")
        return df_scaled, scaler

    try:
        df_scaled[valid_columns] = scaler.fit_transform(df_scaled[valid_columns])
        logger.info("Applied Z-score scaling")
        return df_scaled, scaler
    except ValueError as exc:
        logger.error("Failed Z-score scaling")
        raise ValueError("Failed Z-score scaling") from exc


def apply_one_hot_encoding(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Apply one-hot encoding to selected columns.

    Args:
        df (pd.DataFrame): Input dataset.
        columns (List[str]): Columns to encode.

    Returns:
        pd.DataFrame: Encoded dataset.
    """
    valid_columns = [col for col in columns if col in df.columns]

    if not valid_columns:
        logger.warning("No valid columns for one-hot encoding")
        return df

    try:
        encoded = pd.get_dummies(df, columns=valid_columns, drop_first=True)
        logger.info("Applied one-hot encoding")
        return encoded
    except ValueError as exc:
        logger.error("Failed one-hot encoding")
        raise ValueError("Failed one-hot encoding") from exc


def apply_label_encoding(
    df: pd.DataFrame, columns: List[str]
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Apply label encoding to selected columns.

    Args:
        df (pd.DataFrame): Input dataset.
        columns (List[str]): Columns to encode.

    Returns:
        Tuple[pd.DataFrame, Dict[str, LabelEncoder]]: Encoded dataset and encoders.
    """
    encoders: Dict[str, LabelEncoder] = {}
    df_encoded = df.copy()
    valid_columns = [col for col in columns if col in df.columns]

    if not valid_columns:
        logger.warning("No valid columns for label encoding")
        return df_encoded, encoders

    for column in valid_columns:
        try:
            encoder = LabelEncoder()
            df_encoded[column] = encoder.fit_transform(df_encoded[column].astype(str))
            encoders[column] = encoder
            logger.info(f"Applied label encoding to {column}")
        except ValueError as exc:
            logger.error(f"Failed label encoding for {column}")
            raise ValueError(f"Failed label encoding for {column}") from exc

    return df_encoded, encoders


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for modeling.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with engineered features.
    """
    df_features = df.copy()

    if "valor_total" in df_features.columns:
        df_features["categoria_consumo"] = pd.cut(
            df_features["valor_total"],
            bins=3,
            labels=["Bajo", "Medio", "Alto"],
        )
        logger.info("Created categoria_consumo")
    else:
        logger.warning("Column valor_total not found for categoria_consumo")

    if "tiempo_preparacion" in df_features.columns:
        conditions = [
            df_features["tiempo_preparacion"] <= 20,
            df_features["tiempo_preparacion"] <= 40,
        ]
        choices = ["Normal", "Moderado"]
        df_features["nivel_demora"] = np.select(
            conditions,
            choices,
            default="Critico",
        )
        logger.info("Created nivel_demora")
    else:
        logger.warning("Column tiempo_preparacion not found for nivel_demora")

    if {"cliente_frecuente", "calificacion_cliente"}.issubset(df_features.columns):
        df_features["cliente_premium"] = (
            (df_features["cliente_frecuente"] == "Si")
            & (df_features["calificacion_cliente"] >= 4)
        )
        logger.info("Created cliente_premium")
    else:
        logger.warning("Columns not found for cliente_premium")

    if {"valor_total", "cantidad_productos"}.issubset(df_features.columns):
        promedio = df_features["valor_total"] / df_features["cantidad_productos"]
        promedio = promedio.replace([np.inf, -np.inf], np.nan)
        median_value = float(np.nanmedian(promedio))
        df_features["promedio_compra_cliente"] = promedio.fillna(median_value)
        logger.info("Created promedio_compra_cliente")
    else:
        logger.warning("Columns not found for promedio_compra_cliente")

    return df_features


def apply_pca(df: pd.DataFrame, n_components: int) -> Tuple[pd.DataFrame, PCA]:
    """
    Apply PCA to reduce dimensionality.

    Args:
        df (pd.DataFrame): Input dataset.
        n_components (int): Number of PCA components.

    Returns:
        Tuple[pd.DataFrame, PCA]: Transformed dataset and fitted PCA object.
    """
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)

    try:
        components = pca.fit_transform(df)
        columns = [f"pca_{i + 1}" for i in range(n_components)]
        df_pca = pd.DataFrame(components, columns=columns, index=df.index)
        logger.info("Applied PCA")
        return df_pca, pca
    except ValueError as exc:
        logger.error("Failed PCA")
        raise ValueError("Failed PCA") from exc


def prepare_for_modeling(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare the dataset for modeling.

    Args:
        df (pd.DataFrame): Input dataset.
        target_col (str): Target column name.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Train and test splits.
    """
    if target_col not in df.columns:
        message = f"Target column {target_col} not found"
        logger.error(message)
        raise ValueError(message)

    df_model = df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns])
    y = df_model[target_col]
    X = df_model.drop(columns=[target_col])

    stratify = None
    if y.dtype == "object" or y.dtype.name == "category":
        stratify = y
        encoder = LabelEncoder()
        try:
            y = pd.Series(encoder.fit_transform(y.astype(str)), index=y.index)
            logger.info("Encoded categorical target")
        except ValueError as exc:
            logger.error("Failed to encode target")
            raise ValueError("Failed to encode target") from exc

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
        logger.info("Prepared train test split")
        return X_train, X_test, y_train, y_test
    except ValueError as exc:
        logger.error("Failed train test split")
        raise ValueError("Failed train test split") from exc
