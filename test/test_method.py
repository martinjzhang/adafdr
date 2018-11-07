""" Test for the core code of the method module
"""
import numpy as np
import adafdr.method as md
import adafdr.data_loader as dl

def test_method_init():
    """ test for md.method_init
    """
    p, x, h, n_full, _ = dl.load_2d_bump_slope(n_sample=20000)
    a, mu, sigma, w = md.method_init(p, x, 2, alpha=0.1, n_full=n_full, h=h,
                                     random_state=0, fold_number=0)
    t = md.f_all(x, a, mu, sigma, w)
    gamma = md.rescale_mirror(t, p, 0.1)
    t = t*gamma
    n_rej = np.sum(p < t)
    FDP = np.sum((p < t)*(h == 0))/n_rej
    print('n_rej:', n_rej)
    assert n_rej > 700
    assert FDP < 0.15
    mu_ref1 = np.array([[0.25, 0.25], [0.75, 0.75]], dtype=float)
    mu_ref2 = np.array([[0.75, 0.75], [0.25, 0.25]], dtype=float)
    error = np.min([np.linalg.norm(mu-mu_ref1),
                    np.linalg.norm(mu-mu_ref2)])
    print('error for estimating mu = %0.8f'%error)
    assert error < 0.05
def test_reparametrize():
    """ test for md.reparametrize
    """
    w_init = np.array([0.4, 0.3, 0.3], dtype=float)
    a_init = np.array([2, 0.1], dtype=float)
    mu_init = np.array([[0.2, 0.2], [0.7, 0.7]], dtype=float)
    sigma_init = np.array([[0.1, 0.2], [0.1, 0.1]], dtype=float)
    d = 2
    x_test = np.array([[0.1, 0.2], [0.3, 0.5]], dtype=float)
    a, b, w, mu, sigma = md.reparametrize(a_init, mu_init, sigma_init, w_init, d)
    t_init = md.f_all(x_test, a_init, mu_init, sigma_init, w_init)
    t = md.t_cal(x_test, a, b, w, mu, sigma)
    print('t_init:', t_init)
    print('t', t)
    assert all(np.absolute(t_init-t) < 1e-8)
def test_rescale_mirror():
    """ test for md.rescale_mirror
    """
    p, x, _, _, _ = dl.load_2d_bump_slope(n_sample=2000)
    alpha = 0.1
    t = np.ones([x.shape[0]], dtype=float)
    gamma_grid = np.linspace(1e-4, 0.01, 100)
    alpha_hat = np.zeros([gamma_grid.shape[0]], dtype=float)
    for i in range(gamma_grid.shape[0]):
        alpha_hat[i] = np.sum(p > 1-t*gamma_grid[i])/np.sum(p < t*gamma_grid[i])
    gamma = np.max(gamma_grid[alpha_hat < alpha])
    gamma_test = md.rescale_mirror(t, p, alpha)
    print('gamma_GT', gamma)
    print('gamma_test', gamma_test)
    assert np.absolute(gamma-gamma_test) < 1e-4
def test_method_single_fold():
    """ test for md.method_single_fold
    """
    p, x, h, n_full, _ = dl.load_2d_bump_slope(n_sample=20000)
    n_rej, t, _ = md.method_single_fold(p, x, 2, alpha=0.1, n_full=n_full,
                                        n_itr=100, h=h, fold_number=0, random_state=0)
    FDP = np.sum((p < t)*(h == 0))/n_rej
    print('n_rej:', n_rej)
    assert n_rej > 800
    print('FDP:', n_rej)
    assert FDP < 0.15
def test_preprocess_two_fold():
    """ Test for preprocess_two_fold
    """
    np.random.seed(0)
    x_test_1 = np.random.choice([0, 1, 2, 3], size=300)
    x_test_2 = np.array([0, 1, 2, 3]).reshape([-1, 1])
    temp = np.arange(300)
    p_test = np.ones([300], dtype=float)
    p_test[x_test_1 == 0] = 0.001
    p_test[(x_test_1 == 1)*(temp < 200)] = 0.001
    p_test[(x_test_1 == 2)*(temp < 100)] = 0.001
    _, x_test_new_2 = md.preprocess_two_fold(p_test,
                                             x_test_1.reshape([-1, 1]),
                                             x_test_2,
                                             300, None)
    print('x_test_2', x_test_2)
    print('x_test_new_2', x_test_new_2)
    assert x_test_new_2[0] > 0.75
    assert (x_test_new_2[1] > 0.5) and (x_test_new_2[1] < 0.75)
    assert (x_test_new_2[2] > 0.25) and (x_test_new_2[2] < 0.5)
    assert x_test_new_2[3] < 0.25
def test_adafdr_test():
    """ Test for adafdr_test
    """
    p, x, h, n_full, _ = dl.load_2d_bump_slope(n_sample=20000)
    res = md.adafdr_test(p, x, K=2, alpha=0.1, h=None, n_full=n_full,\
                         n_itr=50, verbose=False, random_state=0,\
                         single_core=True)
    t = res['threshold']
    FDP = np.sum((p < t)*(h == 0))/np.sum(p < t)
    n_rej = np.sum(p < t)
    print('n_rej', n_rej)
    assert n_rej > 700
    print('FDP', FDP)
    assert FDP < 0.12
    