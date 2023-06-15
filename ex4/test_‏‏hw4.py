import numpy as np
import pytest
from hw4 import LogisticRegressionGD

from hw4 import norm_pdf
from hw4 import gmm_pdf
from hw4 import sigmoid




# @pytest.fixture
# def example_data():


def test_sigmoid():
    # Test sigmoid function with scalar inputs
    assert sigmoid(0) == 0.5
    assert sigmoid(-5) == pytest.approx(0.00669285092428)
    assert sigmoid(5) == pytest.approx(0.99330714907571)

    # Test sigmoid function with array inputs
    x = np.array([-2, -1, 0, 1, 2])
    expected_output = np.array([0.11920292202211755, 0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823])
    np.testing.assert_allclose(sigmoid(x), expected_output, atol=1e-8)

    # Test sigmoid function with column vector input
    x_column = np.array([[-2], [-1], [0], [1], [2]])
    expected_output_column = np.array([[0.11920292202211755], [0.2689414213699951], [0.5], [0.7310585786300049], [0.8807970779778823]])
    np.testing.assert_allclose(sigmoid(x_column), expected_output_column, atol=1e-8)

def test_norm_pdf():
    x = np.array([1, 2, 3, 0, -1, -2, -3])
    mu = 0
    sigma = 1
    expected_output = np.array([0.24197072, 0.05399097, 0.00443185, 0.39894228, 0.24197072, 0.05399097, 0.00443185])
    output = norm_pdf(x, mu, sigma)
    np.testing.assert_allclose(output, expected_output, atol=1e-5)
    for _ in range(1000):
        cur_data = np.random.uniform(low=-4, high=4)
        cur_mu = np.random.uniform(low=-4, high=4)
        cur_sigma = np.random.uniform(low=0.0001, high=4)
        output = norm_pdf(cur_data,cur_mu,cur_sigma)
        assert output >= 0, f"{output} negative! \n data:{cur_data}, mu: {cur_mu},sigma:{cur_sigma} "

def test_gmm_pdf():
    # Test case 1
    data = 2
    weights = [0.3, 0.7]
    mus = [1, 2]
    sigmas = [0.5, 0.8]
    expected_pdf = 0.3814690752591664

    result = gmm_pdf(data, weights, mus, sigmas)
    assert np.isclose(result, expected_pdf)

    # Test case 2
    data = -1
    weights = [0.4, 0.6]
    mus = [-2, 0]
    sigmas = [1.2, 0.5]
    expected_pdf = 0.15875978495259319

    result = gmm_pdf(data, weights, mus, sigmas)
    assert np.isclose(result, expected_pdf)