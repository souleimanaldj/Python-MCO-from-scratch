import numpy as np
import pandas as pd
from typing import Tuple
from Linear_Regression_LinearRegression import LinearRegression

class Result:
    """
    Stores a fitted linear regression model with the true and predicted
    values for both training and test sets. Computes evaluation metrics such as
    R², MSE, and RMSE for model assessment.
    """

    def __init__(
        self,
        model: LinearRegression,
        y_train: pd.DataFrame,
        y_pred_train: np.ndarray,
        y_test: pd.DataFrame,
        y_pred_test: np.ndarray
    ) -> None:
        """
        Initializes the Result instance.

        Parameters:
            model : LinearRegression.
            y_train : pd.DataFrame.
            y_pred_train : np.ndarray.
            y_test : pd.DataFrame.
            y_pred_test : np.ndarray.

        Returns:
            None.
        """

        self.model = model
        self.R2 = None
        self.y_train = y_train.values
        self.y_pred_train = y_pred_train
        self.y_test = y_test.values
        self.y_pred_test = y_pred_test

    def get_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the R².

        Parameters:
            y_true : np.ndarray.
            y_pred : np.ndarray.

        Returns:
            R2 : float.
        """

        scr = float(np.sum((y_true - y_pred) ** 2))
        sct = float(np.sum((y_true - np.mean(y_true)) ** 2))
        R2 = 1 - (scr / sct)
        return R2

    def calculate_error(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[float, float]:
        """
        Computes the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

        Parameters:
            y_true : np.ndarray.
            y_pred : np.ndarray.

        Returns:
            mse : float.
            rmse : float.
        """

        mse = float(np.mean((y_true - y_pred)**2))
        rmse = float(np.sqrt(mse))
        return mse, rmse

    def results(self) -> str:
        """
        Generates a summary of the model performance, including
        coefficients, R², MSE, and RMSE for both training and test sets.

        Returns:
            A summary string containing:
            - coefficients,
            - train R² and test R²,
            - train MSE and RMSE,
            - test MSE and RMSE.
        """

        r2_train = self.get_r2(self.y_train, self.y_pred_train)
        r2_test = self.get_r2(self.y_test, self.y_pred_test)

        mse_train, rmse_train = self.calculate_error(
            self.y_train, self.y_pred_train
        )
        mse_test, rmse_test = self.calculate_error(
            self.y_test, self.y_pred_test
        )

        return (
            f"Model coefficients: {self.model.to_dict()}.\n"
            f"Train R²: {round(r2_train, 3)}.\n"
            f"Test R²: {round(r2_test, 3)}.\n"
            f"Train RMSE: {round(rmse_train, 3)} (MSE={round(mse_train, 3)}).\n"
            f"Test RMSE: {round(rmse_test, 3)} (MSE={round(mse_test, 3)})."
        )
