from copy import copy
import numpy as np
import pytest

from pyfm import FM


@pytest.fixture()
def setup(nrow=1000, ncol=10, seed=23098472):
    np.random.seed(seed)
    
    NUM_VAL = 2 * nrow

    # random normal values
    values = np.random.normal(0, 1, NUM_VAL)
    rows = np.concatenate((
        np.arange(nrow), np.arange(nrow)
    ))
    cols = np.concatenate((
        np.repeat(0, NUM_VAL/2), np.repeat(1, NUM_VAL/2)
    ))
    y = np.random.normal(0, 1, nrow) / 10 + 2 * values[:int(NUM_VAL/2)] - \
        1 * values[int(NUM_VAL/2):NUM_VAL]

    y_ind = np.arange(nrow)
    K = 3
    return values, rows, cols, y, y_ind, K


@pytest.fixture()
def params1():
    opt_params_sgd = {
        "optimizer": "sgd",
        "minibatch": 128,
        "n_outer": 10000,
        "eta" : 0.001,
        "lambda" : 0,
        "eps": 1e-8
    }
    return opt_params_sgd


@pytest.fixture()
def params_adam(params1):
    opt_params = params1
    opt_params = opt_params.copy()
    opt_params['minibatch'] = 128
    opt_params['n_outer'] = 10000
    opt_params['optimizer'] = 'adam'
    opt_params['eta'] = .001
    opt_params['lambda'] = .001
    return opt_params


@pytest.fixture()
def params_adagrad(params1):
    opt_params = params1
    opt_params = opt_params.copy()
    opt_params['optimizer'] = 'adagrad'
    opt_params['eta'] = .05
    return opt_params


def test_sgd_main_effects(setup, params1, seed=238423):
    values, rows, cols, y, y_ind, K = setup

    np.random.seed(seed)
    
    fm = FM(K, params1, True)
    fm.fit(values, rows, cols, y)
    print(fm.f_beta0)
    print(fm.f_beta)
    print(fm.f_v)
    np.testing.assert_almost_equal(fm.f_beta0, 0, 2)
    np.testing.assert_almost_equal(fm.f_beta[0], 2, 2)
    np.testing.assert_almost_equal(fm.f_beta[1], -1, 2)

    # assert that too few iterations produces high in-sample loss
    loss1 = fm.loss(values, rows, cols, y)
    params1['n_outer'] = 10
    fm.fit(values, rows, cols, y)
    loss2 = fm.loss(values, rows, cols, y)
    assert loss1 < loss2


def test_adam_main_effects(setup, params_adam, seed=238423):
    np.random.seed(seed)    
    para = params_adam    
    values, rows, cols, y, y_ind, K = setup
    
    fm = FM(K, para, True)
    fm.fit(values, rows, cols, y)
    print(fm.f_beta0)
    print(fm.f_beta)
    print(fm.f_v)
    np.testing.assert_almost_equal(fm.f_beta0, 0, 2)
    np.testing.assert_almost_equal(fm.f_beta[0], 2, 2)
    np.testing.assert_almost_equal(fm.f_beta[1], -1, 2)

    # assert that too few iterations produces high in-sample loss
    loss1 = fm.loss(values, rows, cols, y)
    para['n_outer'] = 10
    fm.fit(values, rows, cols, y)
    loss2 = fm.loss(values, rows, cols, y)
    assert loss1 < loss2


def test_adagrad_main_effects(setup, params_adagrad, seed=238423):
    np.random.seed(seed)    
    para = params_adagrad    
    values, rows, cols, y, y_ind, K = setup
    
    fm = FM(K, para, True)
    fm.fit(values, rows, cols, y)
    print(fm.f_beta0)
    print(fm.f_beta)
    print(fm.f_v)
    np.testing.assert_almost_equal(fm.f_beta0, 0, 2)
    np.testing.assert_almost_equal(fm.f_beta[0], 2, 2)
    np.testing.assert_almost_equal(fm.f_beta[1], -1, 2)

    # assert that too few iterations produces high in-sample loss
    loss1 = fm.loss(values, rows, cols, y)
    para['n_outer'] = 10
    fm.fit(values, rows, cols, y)
    loss2 = fm.loss(values, rows, cols, y)
    assert loss1 < loss2

