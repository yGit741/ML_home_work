import pytest
from hw3 import conditional_independence

@pytest.fixture
def distribution():
    return conditional_independence()


def test_variable_probabilities(distribution):
    assert sum(distribution.X.values()) == pytest.approx(1.0)
    assert sum(distribution.Y.values()) == pytest.approx(1.0)
    assert sum(distribution.C.values()) == pytest.approx(1.0)

def test_joint_probabilities(distribution):
    assert sum(distribution.X_Y.values()) == pytest.approx(1.0)
    assert sum(distribution.X_C.values()) == pytest.approx(1.0)
    assert sum(distribution.Y_C.values()) == pytest.approx(1.0)

def test_joint_conditional_probabilities(distribution):
    assert sum(distribution.X_Y_C.values()) == pytest.approx(1.0)



