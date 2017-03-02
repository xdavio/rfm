from scipy.sparse import coo_matrix
import numpy as np
import pytest

from pyfm import FM

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
        "optimizer": "adam",
        "minibatch": 100,
        "n_outer": 30000,
        "eta" : 0.001,
        "lambda" : 0,
        "eps": 1e-8
    }
    K = 4
    return K, opt_params_sgd

def test_sgd_int(data, params1):
    values, rows, cols, y = data
    K, opt_params = params1
    
    fm = FM(K, opt_params)
    y_mean = y.mean()
    y_sd = y.std()
    y = (y - y_mean) / y_sd
    fm.fit(values, rows, cols, y)

    print(fm.f_beta0)
    print(fm.f_beta[:3])
    print(y_mean, y_sd)
    print(fm.f_beta0 * y_sd + y_mean)
    print(fm.f_beta[:3] * y_sd)
    print(fm.f_v[1,:].dot(fm.f_v[2,:]) * y_sd)
    #np.testing.assert_almost_equal(fm.f_beta0, 0, 1)
    np.testing.assert_almost_equal(fm.f_beta[0], 1, 1)
    np.testing.assert_almost_equal(fm.f_beta[1], 2, 1)
    np.testing.assert_almost_equal(fm.f_beta[2], -1, 1)    

