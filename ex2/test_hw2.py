import numpy as np
import pandas as pd
import pytest
import  hw2
from scipy.stats import entropy

@pytest.fixture()
def raw_data_fixtures():
    data = pd.read_csv('agaricus-lepiota.csv')
    X, y = data.drop('class', axis=1), data['class']
    X = np.column_stack([X, y])
    return X

@pytest.fixture()
def splited_data_fixtures(raw_data_fixtures):
    data = raw_data_fixtures()
    X_train, X_test = train_test_split(X, random_state=99)
    return X_train, X_test

def test_calc_gini():
    # Test case 1
    data1 = [[1, 'A'], [2, 'B'], [3, 'A'], [4, 'B'], [5, 'A']]
    expected_gini1 = 0.48
    assert np.isclose(hw2.calc_gini(data1), expected_gini1, atol=1e-4)

    # Test case 2
    data2 = [[1, 'A'], [2, 'A'], [3, 'A'], [4, 'A'], [5, 'A']]
    expected_gini2 = 0.0
    assert np.isclose(hw2.calc_gini(data2), expected_gini2, atol=1e-4)

    # Test case 3
    data3 = [[1, 'A'], [2, 'B'], [3, 'C'], [4, 'D'], [5, 'E']]
    expected_gini3 = 0.8
    assert np.isclose(hw2.calc_gini(data3), expected_gini3, atol=1e-4)


def test_calc_entropy():
    # Test case 1
    data1 = [[1, 'A'], [2, 'B'], [3, 'A'], [4, 'B'], [5, 'A']]
    expected_entropy1 = 0.971
    assert np.isclose(hw2.calc_entropy(data1), expected_entropy1, atol=1e-3)

    # Test case 2
    data2 = [[1, 'A'], [2, 'A'], [3, 'A'], [4, 'A'], [5, 'A']]
    expected_entropy2 = 0.0
    assert np.isclose(hw2.calc_entropy(data2), expected_entropy2, atol=1e-4)

    # Test case 3
    data3 = [[1, 'A'], [2, 'B'], [3, 'C'], [4, 'D'], [5, 'E']]
    expected_entropy3 = 2.322
    assert np.isclose(hw2.calc_entropy(data3), expected_entropy3, atol=1e-3)


def test_goodness_of_split():
    # Test with Gini impurity
    impurity_func = hw2.calc_gini
    goodness, groups = hw2.goodness_of_split(X_train, 0, impurity_func, gain_ratio=False)
    assert isinstance(goodness, float), "Goodness of split should be a float"
    assert isinstance(groups, dict), "Groups should be a dictionary"

    # Test with Gain Ratio
    impurity_func = hw2.calc_entropy
    goodness, groups = hw2.goodness_of_split(X_train, 1, impurity_func, gain_ratio=True)
    assert isinstance(goodness, float), "Goodness of split should be a float"
    assert isinstance(groups, dict), "Groups should be a dictionary"
