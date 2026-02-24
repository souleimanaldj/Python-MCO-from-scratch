import numpy as np
from sklearn.datasets import make_regression
from Linear_Regression_DataSet import DataSet
from Linear_Regression_LinearRegression import LinearRegression
from Linear_Regression_Result import Result

def main(X: np.ndarray,
         y: np.ndarray,
         features_name: list[str],
         target_name: list[str]) -> None:
    """
    Runs the full regression on user-provided data:
    - builds the dataset object,
    - adds an intercept column,
    - converts data into train/test DataFrames,
    - fits a linear regression model,
    - generates predictions,
    - computes evaluation metrics and prints a summary.

    Parameters:
        X : numpy.ndarray of shape (n_samples, n_features).
        y : numpy.ndarray of shape (n_samples,) or (n_samples, 1).
        features_name : list of str.
        target_name : list of str.

    Returns:
        None.
    """

    data = DataSet(X, y, features_name, target_name)
    data.add_intercept()


    X_train, y_train, X_test, y_test = data.turn_into_dataframe()

    model = LinearRegression(data.features_name)
    y_pred_train = model.fit_model(X_train, y_train)
    y_pred_test = model.predict_model(X_test)

    result = Result(model, y_train, y_pred_train, y_test, y_pred_test)
    print(result.results())



X, y = make_regression(
        n_samples=15000,
        n_features=3,
        random_state=42,
        noise=50
)
features_name = ["X1", "X2", "X3"]
target_name = ["y"]

main(X, y, features_name, target_name)
