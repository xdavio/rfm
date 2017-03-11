from scipy.sparse import coo_matrix
import numpy as np
import pytest
import logging

from pyfm import FM

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture()
def data(nrow=5000, ncol=10, seed=4832):
    np.random.seed(seed)

    X = np.random.normal(0, 1, (nrow, ncol))
    coef = np.zeros(ncol)
    coef[0] = 1
    coef[1] = 2
    coef[2] = -1
    int12 = .5
    y = np.random.normal(0, 1, nrow) / 1000 + \
           X.dot(coef) + X[:, 1]*X[:, 2]*int12

    X = coo_matrix(X)
    return X.data, X.row, X.col, y


@pytest.fixture()
def params1():
    opt_params_sgd = {
        "optimizer": "sgd",
        "minibatch": 100,
        "n_outer": 20000,
        "eta" : 0.001,
        "lambda" : 0.0001,
        "eps": 1e-8
    }
    K = 10
    return K, opt_params_sgd


@pytest.fixture()
def params_adam(params1):
    K, opt_params = params1
    opt_params = opt_params.copy()
    opt_params['optimizer'] = 'adam'
    opt_params['eta'] = .0001
    return K, opt_params.copy()


@pytest.fixture()
def params_adagrad(params1):
    K, opt_params = params1
    opt_params = opt_params.copy()
    opt_params['optimizer'] = 'adagrad'
    opt_params['eta'] = .1
    return K, opt_params.copy()


def test_sgd_int(data, params1):
    log = logging.getLogger('test_sgd_int')
    values, rows, cols, y = data
    K, opt_params = params1
    
    fm = FM(K, opt_params, standardize=True)
    fm.fit(values, rows, cols, y)

    est_int12 = fm.f_v[1, :].dot(fm.f_v[2, :])

    log.debug('intercept: {}'.format(fm.f_beta0))
    log.debug('first 3 main effects: {}'.format(fm.f_beta[:3]))
    log.debug('remaining main effects: {}'.format(fm.f_beta[3:10]))
    log.debug('interaction effect of 1+2: {}'.format(est_int12))

    np.testing.assert_almost_equal(fm.f_beta0, 0, 2)
    np.testing.assert_almost_equal(fm.f_beta[0], 1, 2)
    np.testing.assert_almost_equal(fm.f_beta[1], 2, 2)
    np.testing.assert_almost_equal(fm.f_beta[2], -1, 2)


def test_adam_int(data, params_adam):
    log = logging.getLogger('test_sgd_int')
    values, rows, cols, y = data
    K, opt_params = params_adam
    
    fm = FM(K, opt_params, standardize=True)
    fm.fit(values, rows, cols, y)

    est_int12 = fm.f_v[1, :].dot(fm.f_v[2, :])

    log.debug('intercept: {}'.format(fm.f_beta0))
    log.debug('first 3 main effects: {}'.format(fm.f_beta[:3]))
    log.debug('remaining main effects: {}'.format(fm.f_beta[3:10]))
    log.debug('interaction effect of 1+2: {}'.format(est_int12))

    np.testing.assert_almost_equal(fm.f_beta0, 0, 2)
    np.testing.assert_almost_equal(fm.f_beta[0], 1, 2)
    np.testing.assert_almost_equal(fm.f_beta[1], 2, 2)
    np.testing.assert_almost_equal(fm.f_beta[2], -1, 2)
    

def test_adagrad_int(data, params_adagrad):
    log = logging.getLogger('test_sgd_int')
    values, rows, cols, y = data
    K, opt_params = params_adagrad
    
    fm = FM(K, opt_params, standardize=True)
    fm.fit(values, rows, cols, y)

    est_int12 = fm.f_v[1, :].dot(fm.f_v[2, :])

    log.debug('intercept: {}'.format(fm.f_beta0))
    log.debug('first 3 main effects: {}'.format(fm.f_beta[:3]))
    log.debug('remaining main effects: {}'.format(fm.f_beta[3:10]))
    log.debug('interaction effect of 1+2: {}'.format(est_int12))

    np.testing.assert_almost_equal(fm.f_beta0, 0, 2)
    np.testing.assert_almost_equal(fm.f_beta[0], 1, 2)
    np.testing.assert_almost_equal(fm.f_beta[1], 2, 2)
    np.testing.assert_almost_equal(fm.f_beta[2], -1, 2)
    
    
