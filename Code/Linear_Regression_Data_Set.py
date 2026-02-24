import numpy as np
import pandas as pd
from typing import List, Tuple


class DataSet:
    """
    Stores a feature matrix (X), a target vector (y), and their column names.
    Adds an intercept to the feature matrix and converts the data into
    pandas DataFrames with a train/test split.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        features_name: List[str],
        target_name: List[str]
    ) -> None:
        """
        Initializes the DataSet instance and reshapes the target vector
        into a column vector.

        Parameters:
            X : np.ndarray.
            y : np.ndarray.
            features_name : List[str].
            target_name : List[str].

        Returns:
            None.
        """

        self.X = X
        self.y = y.reshape(-1, 1)
        self.features_name = features_name
        self.target_name = target_name

    def add_intercept(self) -> None:
        """
        Raises an error if there is a dimension mismatch between X and y
        Adds a column of ones as the first column of the feature matrix.
        Also adds the name "intercept" at the beginning of the feature names list.

        Returns:
            None.
        """
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        intercept = np.ones((self.X.shape[0], 1))
        self.X = np.concatenate((intercept, self.X), axis=1)
        self.features_name = ["intercept"] + self.features_name

    def turn_into_dataframe(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Converts the feature matrix (X) and the target vector (y)
        into four pandas DataFrames using an 80/20 train–test split.

        Returns:
            X_train : pd.DataFrame.
            y_train : pd.DataFrame.
            X_test : pd.DataFrame.
            y_test : pd.DataFrame.
        """

        df_X = pd.DataFrame(self.X, columns=self.features_name)
        df_y = pd.DataFrame(self.y, columns=self.target_name)

        cutoff = int(len(self.X) * 0.8)
        X_train = df_X.iloc[:cutoff]
        y_train = df_y.iloc[:cutoff]
        X_test = df_X.iloc[cutoff:]
        y_test = df_y.iloc[cutoff:]

        return X_train, y_train, X_test, y_test
