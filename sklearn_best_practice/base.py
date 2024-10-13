from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)


class BaseModel:
    """
    Base class for all sklearn-like format models

    Parameters:
        model (Any): The model object
    """

    def __init__(self, model: Any, config: None | Dict[str, Any]) -> None:
        self.model = model(**config) if config is not None else model()

    def train(
        self,
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        x_te: np.ndarray
    ) -> np.ndarray:
        """
        Train the model

        Parameters:
            X (np.ndarray): The input data
            y (np.ndarray): The target

        Returns:
            np.ndarray: The predicted labels
        """
        self.model.fit(x_tr, y_tr)
        y_pred = self.model.predict(x_te)

        return y_pred

    def evaluate(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        issue: str = "reg"
    ) -> float:
        """
        Evaluate the model

        Parameters:
            y_pred (np.ndarray): The predicted labels
            y_true (np.ndarray): The true labels
            issue (str): The issue type (regression or classification)

        Returns:
            clf: accuracy, auc
            reg: mse, r-squared
        """
        if issue == "reg":
            print("Regression issue")
            print(
                f"MSE: {root_mean_squared_error(y_true, y_pred)}, R2: {r2_score(y_true, y_pred)}"
            )

        elif issue == "clf":
            print("Classification issue")
            print(
                f"Accuracy: {accuracy_score(y_true, y_pred)}, Recall: {recall_score(y_true, y_pred, average='macro')}"
            )

        else:
            raise ValueError("Issue type must be either 'reg' or 'clf'")
