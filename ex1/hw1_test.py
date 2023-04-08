import hw1
import pytest
import pandas as pd
import numpy as np

#################################### FIXTURES ####################################

@pytest.fixture()
def raw_data_fixtures():
    df = pd.read_csv('data.csv')
    X = df['sqft_living'].values
    y = df['price'].values
    return (df, X, y)


@pytest.fixture()
def splitted_data_fixtures(raw_data_fixtures):
    df, X, y = raw_data_fixtures
    np.random.seed(42)
    indices = np.random.permutation(X.shape[0])
    idx_train, idx_val = indices[:int(0.8 * X.shape[0])], indices[int(0.8 * X.shape[0]):]
    X_train, X_val = hw1.preprocess(X[idx_train], X[idx_val])
    y_train, y_val = hw1.preprocess(y[idx_train], y[idx_val])
    return  X_train, X_val, y_train, y_val

@pytest.fixture()
def splitted_data_bias_fixtures(splitted_data_fixtures):
    X_train, X_val, y_train, y_val = splitted_data_fixtures
    return hw1.apply_bias_trick(X_train), hw1.apply_bias_trick(X_val), y_train, y_val


##################################### TESTS #####################################


def test_preprocess(raw_data_fixtures):
    df, X, y = raw_data_fixtures
    X, y = hw1.preprocess(X, y)
    assert all(np.logical_and(-1 <= X, X <= 1)), "X is not between -1 to 1"
    assert all(np.logical_and(-1 <= y, y <= 1)), "y is not between -1 to 1"


def test_apply_bias_trick():
    vec_1d_col_numpy = np.ones(3).reshape(-1,1)
    vec_1d_row_numpy = np.ones(3)
    matrix_numpy = np.ones(6).reshape(3,2)
    assert np.array_equal(hw1.apply_bias_trick(vec_1d_col_numpy), np.array([[1,1],[1,1],[1,1]]))
    assert np.array_equal(hw1.apply_bias_trick(vec_1d_row_numpy), np.array([[1,1],[1,1],[1,1]]))
    assert np.array_equal(hw1.apply_bias_trick(matrix_numpy), np.array([[1,1,1],[1,1,1],[1,1,1]]))


def test_compute_cost(splitted_data_bias_fixtures):
    X_train, X_val, y_train, y_val = splitted_data_bias_fixtures
    assert isinstance(hw1.compute_cost(X_train, y_train, [1,1]), np.float64)


def test_mse_partial_derivative_vectorized(splitted_data_bias_fixtures):
    X_train, X_val, y_train, y_val = splitted_data_bias_fixtures
    assert isinstance(hw1._mse_partial_derivative_vectorized([1,2],1,X_train, y_train), np.float64)


def test_gradient_descent2():
    # create a simple dataset
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([3, 7, 11])
    theta = np.array([0, 0])
    alpha = 0.01
    num_iters = 100

    # run gradient descent
    theta, J_history = hw1.gradient_descent(X, y, theta, alpha, num_iters)

    # check that the output has the expected shape
    assert theta.shape == (2,)
    assert len(J_history) == num_iters

    # check that the final parameters are close to the expected values
    assert np.allclose(theta, np.array([2, 1]))

    # check that the loss value is decreasing with each iteration
    for i in range(1, len(J_history)):
        assert J_history[i] < J_history[i-1]