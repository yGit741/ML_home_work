import hw1
import pytest
import pandas as pd
import numpy as np


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
    X_train, X_val = X[idx_train], X[idx_val]
    y_train, y_val = y[idx_train], y[idx_val]
    return  X_train, X_val, y_train, y_val


def test_preprocess(raw_data_fixtures):
    df, X, y = raw_data_fixtures
    X, y = hw1.preprocess(X, y)
    assert all(np.logical_and(-1 <= X, X <= 1)), "X is not between -1 to 1"
    assert all(np.logical_and(-1 <= y, y <= 1)), "y is not between -1 to 1"


def test_apply_bias_trick(X):
    pass