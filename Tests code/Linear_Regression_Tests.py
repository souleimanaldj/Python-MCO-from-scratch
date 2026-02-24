from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from Linear_Regression_DataSet import DataSet
from Linear_Regression_LinearRegression import LinearRegression
from Linear_Regression_Result import Result


#Class DataSet
def test_dataset_initializes_properly():
    X = np.random.randn(5, 2)
    y = np.random.randn(5)
    features_name = ["X1", "X2"]
    target_name = ["y"]

    ds = DataSet(X, y, features_name, target_name)

    assert ds.X.shape == (5, 2)
    assert ds.y.shape == (5, 1)
    assert ds.features_name == ["X1", "X2"]
    assert ds.target_name == ["y"]


def test_raises_error_on_dimension_mismatch():
    X = np.random.randn(100, 2)
    y = np.random.randn(99)

    try:
        ds = DataSet(X, y, ["X1", "X2"], ["y"])
        ds.add_intercept()
        assert False, "ValueError was not raised"
    except ValueError as error:
        assert str(error) == "X and y must have the same number of rows"


def test_add_intercept():
    X = np.random.randn(5, 2)
    y = np.random.randn(5)

    ds = DataSet(X, y, ["X1", "X2"], ["y"])
    ds.add_intercept()

    assert ds.X.shape == (5, 3)
    assert np.all(ds.X[:, 0] == 1)
    assert ds.features_name[0] == "intercept"


def test_turn_into_dataframe():
    X = np.random.randn(10, 2)
    y = np.random.randn(10)

    ds = DataSet(X, y, ["X1", "X2"], ["y"])
    X_train, y_train, X_test, y_test = ds.turn_into_dataframe()

    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2
    assert type(X_train) == pd.DataFrame
    assert type(X_test) == pd.DataFrame
    assert type(y_train) == pd.DataFrame
    assert type(y_test) == pd.DataFrame




#Class LinearRegression
def test_linearregression_initializes_properly():
    model = LinearRegression(["X1", "X2"])

    assert model.features_name == ["X1", "X2"]
    assert model.coeficients is None


def test_fit_model_raises_error_when_matrix_not_invertible():
    X = pd.DataFrame([[1, 2], [2, 4]], columns=["X1", "X2"])
    y = pd.DataFrame([[3], [6]], columns=["y"])

    model = LinearRegression(["X1", "X2"])

    try:
        model.fit_model(X, y)
        assert False, "np.linalg.LinAlgError was not raised"
    except np.linalg.LinAlgError:
        assert True


def test_fit_model_coefficients():
    X = pd.DataFrame([[1,2], [1,3], [2,0]], columns=["X1", "X2"])
    y = pd.DataFrame([[0], [-1], [4]], columns=["y"])

    model = LinearRegression(["X1", "X2"])
    model.fit_model(X, y)

    assert model.coeficients.shape == (2, 1)
    assert type(model.coeficients) == np.ndarray
    assert round(model.coeficients[0,0],0) == 2
    assert round(model.coeficients[1,0],0) == -1


def test_fit_model_prediction():
    X = pd.DataFrame([[1,2], [1,3], [2,0]], columns=["X1", "X2"])
    y = pd.DataFrame([[0], [-1], [4]], columns=["y"])

    model = LinearRegression(["X1", "X2"])
    y_pred = model.fit_model(X, y)

    assert y_pred.shape == (3, 1)
    assert type(y_pred) == np.ndarray
    for i in range(len(y_pred)):
        assert round(y_pred[i,0], 0) == y.iloc[i,0]


def test_predict_model():
    X = pd.DataFrame([[1,2], [1,3], [2,0]], columns=["X1", "X2"])
    y = pd.DataFrame([[0], [-1], [4]], columns=["y"])

    model = LinearRegression(["X1", "X2"])
    model.fit_model(X, y)
    y_pred = model.predict_model(X)

    assert y_pred.shape == (3, 1)
    assert type(y_pred) == np.ndarray


def test_predict_before_fit_raises_error():
    X = pd.DataFrame([[1,2], [1,3], [2,0]], columns=["X1", "X2"])
    model = LinearRegression(["X1", "X2"])

    try:
        model.predict_model(X)
        assert False, "ValueError was not raised"
    except ValueError as error:
        assert str(error) == "Model has not been fitted yet"


def test_to_dict():
    X = pd.DataFrame([[1,2], [1,3], [2,0]], columns=["X1", "X2"])
    y = pd.DataFrame([[0], [-1], [4]], columns=["y"])

    model = LinearRegression(["X1", "X2"])
    model.fit_model(X, y)
    d = model.to_dict()

    assert type(d) == dict
    assert set(d.keys()) == {"X1", "X2"}
    for v in d.values():
        assert type(v) == float
    assert d["X1"] == 2
    assert d["X2"] == -1




#Class Result
def test_result_initializes_properly():
    model = LinearRegression(["X1", "X2"])
    y_train = pd.DataFrame([[1], [2], [3]])
    y_pred_train = np.array([[1], [2], [3]])
    y_test = pd.DataFrame([[4], [5]])
    y_pred_test = np.array([[4], [5]])

    result = Result(model, y_train, y_pred_train, y_test, y_pred_test)

    assert type(result.model) == LinearRegression
    assert (result.y_train == y_train.values).all()
    assert (result.y_pred_train == y_pred_train).all()
    assert (result.y_test == y_test.values).all()
    assert (result.y_pred_test == y_pred_test).all()


def test_get_r2_correct_calculation():
    y_true = np.array([1, 2, 3]).reshape(-1, 1)
    y_pred = np.array([1, 2, 3]).reshape(-1, 1)

    result = Result(None, pd.DataFrame(y_true), y_pred, pd.DataFrame(y_true), y_pred)
    r2 = result.get_r2(y_true, y_pred)

    assert r2 == 1.0


def test_get_r2_correct_value():
    X, y = make_regression(n_samples=1000, n_features=2, noise=1, random_state=42)
    X = pd.DataFrame(X, columns=["X1", "X2"])
    y = pd.DataFrame(y, columns=["y"])

    model = LinearRegression(["X1", "X2"])
    y_pred = model.fit_model(X, y)
    result = Result(model, y, y_pred, y, y_pred)
    R2 = result.get_r2(y.values, y_pred)

    assert 0 <= R2 <= 1


def test_calculate_error():
    y_true = np.array([1, 2, 3]).reshape(-1, 1)
    y_pred = np.array([1, 2, 3]).reshape(-1, 1)

    result = Result(None, pd.DataFrame(y_true), y_pred, pd.DataFrame(y_true), y_pred)
    mse, rmse = result.calculate_error(y_true, y_pred)

    assert mse == 0.0
    assert rmse == 0.0


def test_results():
    X, y = make_regression(n_samples=100, n_features=2, noise=10, random_state=42)
    X = pd.DataFrame(X, columns=["X1", "X2"])
    y = pd.DataFrame(y, columns=["y"])

    model = LinearRegression(["X1", "X2"])
    y_pred = model.fit_model(X, y)
    result = Result(model, y, y_pred, y, y_pred)
    output = result.results()

    assert type(output) == str
    assert "Model coefficients" in output
    assert "Train R²" in output
    assert "Test R²" in output
    assert "Train RMSE" in output
    assert "Test RMSE" in output
