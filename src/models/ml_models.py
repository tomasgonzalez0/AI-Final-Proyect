"""Traditional machine learning model training for FoodVision AI."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.utils.config import MODELS_DIR, RANDOM_STATE
from src.utils.logger import get_logger


logger = get_logger(__name__)


class MLModelTrainer:
    """Trainer for traditional machine learning models."""

    def train_decision_tree(self, X_train: Any, y_train: Any, task: str) -> Any:
        """
        Train a Decision Tree model for classification or regression.

        Args:
            X_train (Any): Training features.
            y_train (Any): Training labels.
            task (str): Task type, classification or regression.

        Returns:
            Any: Fitted Decision Tree model.
        """
        if task == "classification":
            model = DecisionTreeClassifier(random_state=RANDOM_STATE)
            model_name = "decision_tree"
        elif task == "regression":
            model = DecisionTreeRegressor(random_state=RANDOM_STATE)
            model_name = "decision_tree"
        else:
            message = "Invalid task for decision tree"
            logger.error(message)
            raise ValueError(message)

        try:
            logger.info(f"Training {model_name} for {task}")
            model.fit(X_train, y_train)
            self._save_model(model, model_name, task)
            return model
        except ValueError as exc:
            logger.error(f"Failed training {model_name} for {task}")
            raise ValueError(f"Failed training {model_name} for {task}") from exc

    def train_random_forest(self, X_train: Any, y_train: Any, task: str) -> Any:
        """
        Train a Random Forest model for classification or regression.

        Args:
            X_train (Any): Training features.
            y_train (Any): Training labels.
            task (str): Task type, classification or regression.

        Returns:
            Any: Fitted Random Forest model.
        """
        if task == "classification":
            model = RandomForestClassifier(random_state=RANDOM_STATE)
            model_name = "random_forest"
        elif task == "regression":
            model = RandomForestRegressor(random_state=RANDOM_STATE)
            model_name = "random_forest"
        else:
            message = "Invalid task for random forest"
            logger.error(message)
            raise ValueError(message)

        try:
            logger.info(f"Training {model_name} for {task}")
            model.fit(X_train, y_train)
            self._save_model(model, model_name, task)
            return model
        except ValueError as exc:
            logger.error(f"Failed training {model_name} for {task}")
            raise ValueError(f"Failed training {model_name} for {task}") from exc

    def train_logistic_regression(self, X_train: Any, y_train: Any) -> Any:
        """
        Train a Logistic Regression model for classification.

        Args:
            X_train (Any): Training features.
            y_train (Any): Training labels.

        Returns:
            Any: Fitted Logistic Regression model.
        """
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        model_name = "logistic_regression"
        task = "classification"

        try:
            logger.info("Training logistic regression")
            model.fit(X_train, y_train)
            self._save_model(model, model_name, task)
            return model
        except ValueError as exc:
            logger.error("Failed training logistic regression")
            raise ValueError("Failed training logistic regression") from exc

    def train_linear_regression(self, X_train: Any, y_train: Any) -> Any:
        """
        Train a Linear Regression model for regression.

        Args:
            X_train (Any): Training features.
            y_train (Any): Training labels.

        Returns:
            Any: Fitted Linear Regression model.
        """
        model = LinearRegression()
        model_name = "linear_regression"
        task = "regression"

        try:
            logger.info("Training linear regression")
            model.fit(X_train, y_train)
            self._save_model(model, model_name, task)
            return model
        except ValueError as exc:
            logger.error("Failed training linear regression")
            raise ValueError("Failed training linear regression") from exc

    def train_knn(self, X_train: Any, y_train: Any, task: str) -> Any:
        """
        Train a KNN model for classification or regression.

        Args:
            X_train (Any): Training features.
            y_train (Any): Training labels.
            task (str): Task type, classification or regression.

        Returns:
            Any: Fitted KNN model.
        """
        if task == "classification":
            model = KNeighborsClassifier()
            model_name = "knn"
        elif task == "regression":
            model = KNeighborsRegressor()
            model_name = "knn"
        else:
            message = "Invalid task for knn"
            logger.error(message)
            raise ValueError(message)

        try:
            logger.info(f"Training {model_name} for {task}")
            model.fit(X_train, y_train)
            self._save_model(model, model_name, task)
            return model
        except ValueError as exc:
            logger.error(f"Failed training {model_name} for {task}")
            raise ValueError(f"Failed training {model_name} for {task}") from exc

    def predict(self, model: Any, X_test: Any) -> np.ndarray:
        """
        Generate predictions using a fitted model.

        Args:
            model (Any): Fitted model.
            X_test (Any): Test features.

        Returns:
            np.ndarray: Predicted values.
        """
        try:
            logger.info("Generating predictions")
            return model.predict(X_test)
        except ValueError as exc:
            logger.error("Failed to generate predictions")
            raise ValueError("Failed to generate predictions") from exc

    def _save_model(self, model: Any, model_name: str, task: str) -> None:
        """
        Save a trained model to disk.

        Args:
            model (Any): Trained model to save.
            model_name (str): Name of the model.
            task (str): Task type.

        Returns:
            None: This function returns None.
        """
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            file_path = MODELS_DIR / f"{model_name}_{task}.pkl"
            joblib.dump(model, file_path)
            logger.info(f"Saved model to {file_path}")
        except OSError as exc:
            logger.error("Failed to save model")
            raise OSError("Failed to save model") from exc
