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


def test_calc_gini():
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


def test_calc_entropy():
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


def test_goodness_of_split(raw_data_fixtures):
    X = raw_data_fixtures
    # Test with Gini impurity
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


def test_calc_node_pred():
    # Create example data
    data = pd.DataFrame({'feature1': [1, 2, 3, 4],
                         'feature2': [5, 6, 7, 8],
                         'label': ['a', 'b', 'a', 'a']})

    # Create a DecisionNode instance
    node = hw2.DecisionNode(data)

    # Call calc_node_pred()
    pred = node.calc_node_pred()

    # Assert the predicted label is the one with the highest count
    assert pred == 'a', "Incorrect node prediction"


def test_split():
    # Create dummy data for testing
    data = pd.DataFrame({'feature1': [1, 2, 3, 4, 5],
                         'feature2': [2, 3, 4, 5, 6],
                         'target': [0, 0, 1, 1, 1]})

    # Create a DecisionNode object
    node = hw2.DecisionNode(data, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False)

    # Call the split function
    node.split(impurity_func='gini')

    # Assert that the number of children nodes created is equal to the number of unique values in the split feature
    assert len(node.children) == len(np.unique(data.iloc[:, node.feature]))

    # Assert that the depth of the children nodes is incremented by 1
    assert all(child.depth == node.depth + 1 for child in node.children)

    # # Assert that the chi value of the children nodes is set to the same value as the parent node
    # assert all(child.chi == node.chi for child in node.children)

    # Assert that the max_depth value of the children nodes is set to the same value as the parent node
    assert all(child.max_depth == node.max_depth for child in node.children)

    # Assert that the gain_ratio value of the children nodes is set to the same value as the parent node
    assert all(child.gain_ratio == node.gain_ratio for child in node.children)