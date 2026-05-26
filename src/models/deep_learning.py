"""Deep learning model utilities for FoodVision AI."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger


logger = get_logger(__name__)


def _import_tf_keras() -> tuple[Any, Any, Any]:
    """Import TensorFlow Keras symbols lazily.

    Returns:
        tuple[Any, Any, Any]: (Sequential, Dense, Dropout)

    Raises:
        ModuleNotFoundError: If TensorFlow is not installed.
    """
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, Dropout

        return Sequential, Dense, Dropout
    except ModuleNotFoundError as exc:
        message = (
            "TensorFlow is required for deep learning features. "
            "Install it with: python -m pip install -r requirements-deep-learning.txt"
        )
        logger.error(message)
        raise ModuleNotFoundError(message) from exc


def build_shallow_nn(input_dim: int, output_units: int, task: str) -> Any:
    """
    Build a shallow neural network model.

    Args:
        input_dim (int): Number of input features.
        output_units (int): Number of output units.
        task (str): Task type, classification, binary, or regression.

    Returns:
        Sequential: Compiled Keras model.
    """
    Sequential, Dense, _Dropout = _import_tf_keras()

    model = Sequential()
    model.add(Dense(64, activation="relu", input_dim=input_dim))

    activation, loss = _get_activation_and_loss(task)
    model.add(Dense(output_units, activation=activation))

    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    logger.info("Built shallow neural network")
    return model


def build_deep_nn(input_dim: int, output_units: int, task: str) -> Any:
    """
    Build a deep neural network model.

    Args:
        input_dim (int): Number of input features.
        output_units (int): Number of output units.
        task (str): Task type, classification, binary, or regression.

    Returns:
        Sequential: Compiled Keras model.
    """
    Sequential, Dense, Dropout = _import_tf_keras()

    model = Sequential()
    model.add(Dense(128, activation="relu", input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))

    activation, loss = _get_activation_and_loss(task)
    model.add(Dense(output_units, activation=activation))

    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    logger.info("Built deep neural network")
    return model


def train_neural_network(
    model: Any,
    X_train: Any,
    y_train: Any,
    epochs: int = 50,
    batch_size: int = 32,
) -> Any:
    """
    Train a neural network model.

    Args:
        model (Sequential): Compiled Keras model.
        X_train (Any): Training features.
        y_train (Any): Training labels.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.

    Returns:
        Any: Keras History object.
    """
    try:
        start_time = time.time()
        logger.info("Training neural network")
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0,
        )
        end_time = time.time()
        logger.info(f"Training time seconds: {end_time - start_time:.2f}")
        return history
    except ValueError as exc:
        logger.error("Failed training neural network")
        raise ValueError("Failed training neural network") from exc


def predict_nn(model: Any, X_test: Any) -> np.ndarray:
    """
    Generate predictions using a neural network model.

    Args:
        model (Sequential): Trained Keras model.
        X_test (Any): Test features.

    Returns:
        np.ndarray: Predicted values.
    """
    try:
        logger.info("Generating neural network predictions")
        return model.predict(X_test)
    except ValueError as exc:
        logger.error("Failed neural network prediction")
        raise ValueError("Failed neural network prediction") from exc


def save_model(model: Any, name: str) -> None:
    """
    Save a trained neural network model to disk.

    Args:
        model (Sequential): Trained Keras model.
        name (str): Model name for saving.

    Returns:
        None: This function returns None.
    """
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = MODELS_DIR / f"{name}.h5"
        model.save(file_path)
        logger.info(f"Saved neural network model to {file_path}")
    except OSError as exc:
        logger.error("Failed to save neural network model")
        raise OSError("Failed to save neural network model") from exc


def _get_activation_and_loss(task: str) -> tuple[str, str]:
    """
    Resolve activation and loss for a given task.

    Args:
        task (str): Task type, classification, binary, or regression.

    Returns:
        tuple[str, str]: Activation and loss function names.
    """
    if task == "classification":
        return "softmax", "sparse_categorical_crossentropy"
    if task == "binary":
        return "sigmoid", "binary_crossentropy"
    if task == "regression":
        return "linear", "mse"

    message = "Invalid task for neural network"
    logger.error(message)
    raise ValueError(message)
