import numpy as np
import pandas as pd
from copy import copy

from c_fm import fit_fm


## make some test data to verify that the FM fits
n = 100
p = 30
K = 5
NO_VAL = 100

beta0 = .1
beta = (np.random.random(p) - .5) / 1000
v = (np.random.random((n, K)) - .5) / 1000

values = np.repeat(1, NO_VAL)
rows = np.random.choice(n, NO_VAL)
cols = np.random.choice(p, NO_VAL)

y = np.random.random(n) * 3
y_ind = np.arange(n)

opt_params = {'minibatch': 128,
              'n_outer': 100,
              'eta': .1,
              'lambda': 1}

fitted = fit_fm(beta0, beta, v, opt_params,
                values, rows, cols, y, y_ind, n, p)

print(fitted)
