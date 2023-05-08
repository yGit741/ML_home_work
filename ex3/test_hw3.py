import numpy as np
from hw3 import normal_pdf
from scipy import stats

def test_normal_pdf():
    # Test case 1: Test if the output of the function is a float
    assert isinstance(normal_pdf(1, 0, 1), float)

    # Test case 2: Test if the output of the function is 0.3989422804014327 for x=0, mean=0, std=1
    assert normal_pdf(0, 0, 1) == stats.norm.pdf(0, loc=0, scale=1)


    # Test case 3: Test if the output of the function is 0 for x=0, mean=1, std=1
    assert normal_pdf(0, 1, 1) == 0.24197072451914337

    # Test case 5: Test if the output of the function is None for negative standard deviation
    # assert normal_pdf(1, 0, -1) == None

