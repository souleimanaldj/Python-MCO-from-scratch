import numpy as np
import pandas as pd
from typing import List, Dict

class LinearRegression:
    """
    Stores the feature names and model coefficients, and implements
    Ordinary Least Squares (OLS) regression using the closed-form
    normal equation. Includes methods for fitting the model on training
    data and predicting target values for unseen samples (test data).
    """

    def __init__(self, features_name: List[str]) -> None:
        """
        Initializes the LinearRegression instance.

        Parameters:
            features_name : List[str].

        Returns:
            None.
        """
        self.coeficients = None
        self.features_name = features_name

    def fit_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame
    ) -> np.ndarray:
        """
        Fits the OLS model on the training data
        and returns predictions for the training set.
        Raises an error if the features matrix is not invertible.

        Parameters:
            X_train : pd.DataFrame.
            y_train : pd.DataFrame.

        Returns:
            y_pred : np.ndarray.
        """
        X = X_train.values
        y = y_train.values
        X_t = X.T

        try:
            self.coeficients = np.linalg.inv(X_t @ X) @ X_t @ y
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "Matrix X cannot be inverted."
                "Check for multicollinearity or duplicate features."
            ) from error

        y_pred = X @ self.coeficients
        return y_pred

    def predict_model(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predicts target values for new samples (test data).
        Raises an error if predict_model is called before the model has been fitted.

        Parameters:
            X_test : pd.DataFrame.

        Returns:
            y_pred : np.ndarray.
        """

        if self.coeficients is None:
            raise ValueError("Model has not been fitted yet")

        X = X_test.values
        y_pred = X @ self.coeficients
        return y_pred

    def to_dict(self) -> Dict[str, float]:
        """
        Returns the coefficients as a dictionary mapping
        each feature name to its associated coefficient.

        Returns:
            coefficients : Dict[str, float].
        """

        coefficients = {}
        for i in range(self.coeficients.shape[0]):
            coefficients[self.features_name[i]] = float(
                round(self.coeficients[i, 0], 3)
            )
        return coefficients
Linear_Regression_LinearRegression.txt
