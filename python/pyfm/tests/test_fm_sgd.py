import numpy as np
import pytest

from pyfm import FM


@pytest.fixture()
def setup(nrow=5000, ncol=10, seed=23098472):
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
    y = np.random.normal(0, 1, nrow) / 10 + 2 * values[:(NUM_VAL/2)] - \
        1 * values[(NUM_VAL/2):NUM_VAL]

    y_ind = np.arange(nrow)
    K = 3
    return values, rows, cols, y, y_ind, K


@pytest.fixture()
def params1():
    opt_params_sgd = {
        "optimizer": "sgd",
        "minibatch": 100,
        "n_outer": 10000,
        "eta" : 0.1,
        "lambda" : 0,
        "eps": 1e-8
    }
    return opt_params_sgd


def test_sgd_main_effects(setup, params1, seed=238423):
    values, rows, cols, y, y_ind, K = setup

    np.random.seed(seed)
    
    fm = FM(K, params1)
    fm.fit(values, rows, cols, y)
    print(fm.f_beta0)
    print(fm.f_beta)
    print(fm.f_v)
    np.testing.assert_almost_equal(fm.f_beta0, 0, 1)
    np.testing.assert_almost_equal(fm.f_beta[0], 2, 1)
    np.testing.assert_almost_equal(fm.f_beta[1], -1, 1)

    # assert that too few iterations produces high in-sample loss
    loss1 = fm.loss(values, rows, cols, y)
    params1['n_outer'] = 10
    fm.fit(values, rows, cols, y)
    loss2 = fm.loss(values, rows, cols, y)
    assert loss1 < loss2

