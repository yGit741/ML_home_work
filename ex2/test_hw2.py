import numpy as np
import pandas as pd
import pytest
import  hw2
from scipy.stats import entropy
from sklearn.model_selection import train_test_split

@pytest.fixture()
def raw_data_fixtures():
    data = pd.read_csv('agaricus-lepiota.csv')
    X, y = data.drop('class', axis=1), data['class']
    X = np.column_stack([X, y])
    return X


@pytest.fixture()
def splited_data_fixtures(raw_data_fixtures):
    X = raw_data_fixtures
    X_train, X_test = train_test_split(X, random_state=99)
    return X_train, X_test


def test_calc_gini(raw_data_fixtures):
    # Test case 1
    data1 = np.array([[1, 'A'], [2, 'B'], [3, 'A'], [4, 'B'], [5, 'A']])
    expected_gini1 = 0.48
    assert np.isclose(hw2.calc_gini(data1), expected_gini1, atol=1e-4)

    # Test case 2
    data2 = np.array([[1, 'A'], [2, 'A'], [3, 'A'], [4, 'A'], [5, 'A']])
    expected_gini2 = 0.0
    assert np.isclose(hw2.calc_gini(data2), expected_gini2, atol=1e-4)

    # Test case 3
    data3 = np.array([[1, 'A'], [2, 'B'], [3, 'C'], [4, 'D'], [5, 'E']])
    expected_gini3 = 0.8
    assert np.isclose(hw2.calc_gini(data3), expected_gini3, atol=1e-4)

    # Test case 4
    X = raw_data_fixtures
    expected_gini4 = 0.4995636322379775
    assert expected_gini4 == hw2.calc_gini(X)



def test_calc_entropy(raw_data_fixtures):
    # Test case 1
    data1 = np.array([[1, 'A'], [2, 'B'], [3, 'A'], [4, 'B'], [5, 'A']])
    expected_entropy1 = 0.971
    assert np.isclose(hw2.calc_entropy(data1), expected_entropy1, atol=1e-3)

    # Test case 2
    data2 = np.array([[1, 'A'], [2, 'A'], [3, 'A'], [4, 'A'], [5, 'A']])
    expected_entropy2 = 0.0
    assert np.isclose(hw2.calc_entropy(data2), expected_entropy2, atol=1e-4)

    # Test case 3
    data3 = np.array([[1, 'A'], [2, 'B'], [3, 'C'], [4, 'D'], [5, 'E']])
    expected_entropy3 = 2.322
    assert np.isclose(hw2.calc_entropy(data3), expected_entropy3, atol=1e-3)

    # Test case 4
    X = raw_data_fixtures
    expected_entropy4 = 0.9993703627906085
    assert expected_entropy4 == hw2.calc_entropy(X)



def test_goodness_of_split(raw_data_fixtures):
    X = raw_data_fixtures

    # Test types with Gini impurity
    impurity_func = hw2.calc_gini
    goodness, groups = hw2.goodness_of_split(X, 0, impurity_func, gain_ratio=False)
    assert isinstance(goodness, float), "Goodness of split should be a float"
    assert isinstance(groups, dict), "Groups should be a dictionary"
    assert goodness > 0, "Goodness of split should be positive"

    # Test with Gain Ratio
    impurity_func = hw2.calc_entropy
    goodness, groups = hw2.goodness_of_split(X, 1, impurity_func, gain_ratio=True)
    assert isinstance(goodness, float), "Goodness of split should be a float"
    assert isinstance(groups, dict), "Groups should be a dictionary"
    assert goodness >= 0 and  goodness <= 1, "Gain ratio should be between 0 and 1"

    # Test expected value in compare to X
    expected_goodness_gini = 0.0199596578344422
    expected_goodness_entropy = 0.030727291723502415
    expected_goodness_ratio = 0.0185900525586166
    assert expected_goodness_gini == hw2.goodness_of_split(X, 0, hw2.calc_gini, gain_ratio=False)[0]
    assert expected_goodness_entropy == hw2.goodness_of_split(X, 0, hw2.calc_entropy, gain_ratio=False)[0]
    assert expected_goodness_ratio == hw2.goodness_of_split(X, 0, hw2.calc_entropy, gain_ratio=True)[0]
    assert expected_goodness_ratio == hw2.goodness_of_split(X, 0, hw2.calc_gini, gain_ratio=True)[0]



