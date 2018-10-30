"""Unit test for functions in util.py
"""
import numpy as np
import adafdr.util as util

def test_rank():
    """ Test for util.rank
    """
    x_test = np.array([[0.1, 2, 0.1, 5],
                       [0.1, 10, 5, 18]])
    x_test = x_test.T
    x_rank_continous = util.rank(x_test, continous_rank=True)
    x_rank_discrete = util.rank(x_test, continous_rank=False)
    print(x_rank_continous)
    assert all(x_rank_continous[:, 0] == [0, 2, 1, 3])
    print(x_rank_discrete)
    assert all(x_rank_discrete[:, 0] == [0.5, 2, 0.5, 3])
    