"""Unit test for preprocessing functions
"""
import numpy as np
import adafdr.method as md

def test_feature_preprocess():
    """ Test for method.feature_preprocess
    """
    x_test = np.array([[0.1, 3, 2, 5],
                       [0.1, 10, 5, 18]])
    x_test = md.feature_preprocess(x_test.T)
    print(x_test)
    assert all(x_test[:, 0] == x_test[:, 1])

def test_get_order_discrete():
    """ Test for method.get_order_discrete
    """
    np.random.seed(0)
    x_test = np.random.choice([0, 1, 2, 3], size=300)
    temp = np.arange(300)
    x_val_test = np.array([0, 1, 2, 3])
    p_test = np.ones([300], dtype=float)
    p_test[x_test == 0] = 0.001
    p_test[(x_test == 1)*(temp < 200)] = 0.001
    p_test[(x_test == 2)*(temp < 100)] = 0.001
    x_order = md.get_order_discrete(p_test, x_test, x_val_test)
    print(x_order)
    assert all(x_order == [3, 2, 1, 0])
    x_new_test = md.reorder_discrete(x_test, x_val_test, x_order)
    assert all((x_test == 0) == (x_new_test == 3))
    assert all((x_test == 1) == (x_new_test == 2))
    assert all((x_test == 2) == (x_new_test == 1))
    