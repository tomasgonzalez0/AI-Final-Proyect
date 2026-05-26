"""Data loading utilities for FoodVision AI."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from src.utils.config import DATA_DIR, EXCEL_FILENAME
from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_excel(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load an Excel dataset into a DataFrame.

    Args:
        filepath (Union[str, Path]): Path to the Excel file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    file_path = Path(filepath)

    if not file_path.exists():
        message = (
            "File not found. Place the dataset in the data folder. "
            f"Expected path: {DATA_DIR / EXCEL_FILENAME}"
        )
        logger.error(message)
        raise FileNotFoundError(message)

    try:
        logger.info(f"Loading dataset from {file_path}")
        df = pd.read_excel(file_path)
        logger.info("Dataset loaded")
        return df
    except ValueError as exc:
        logger.error("Failed to read Excel file")
        raise ValueError("Failed to read Excel file") from exc
    except OSError as exc:
        logger.error("File access error while reading Excel file")
        raise OSError("File access error while reading Excel file") from exc


def display_dataset_info(df: pd.DataFrame) -> None:
    """
    Log basic dataset information.

    Args:
        df (pd.DataFrame): Dataset to inspect.

    Returns:
        None: This function returns None.
    """
    logger.info(f"Dataset shape: {df.shape}")
    logger.info("Dataset dtypes")
    logger.info(df.dtypes.to_string())
    logger.info("Dataset head")
    logger.info(df.head(10).to_string(index=False))
    logger.info("Dataset description")
    logger.info(df.describe(include="all").to_string())
