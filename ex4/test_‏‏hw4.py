import numpy as np
import pytest
from hw4 import LogisticRegressionGD

# @pytest.fixture
# def example_data():


def test_sigmoid(example_data):
    theta = np.array([0.5, -0.2])  # Example weight vector
    x = np.array([1, 2])  # Example input vector
    expected_output = 0.7310585786300049  # Expected sigmoid value
    assert sigmoid(theta, x) == pytest.approx(expected_output, abs=1e-6)