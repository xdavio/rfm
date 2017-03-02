import numpy as np

from c_fm import fit_fm, predictfm

OPTIMIZERS = ['adam', 'adagrad', 'sgd']


class FM:
    def __init__(self, K, opt_params):
        if opt_params['optimizer'] not in OPTIMIZERS:
            raise ValueError('Supplied optimizer is not one of ' +
                             ', '.join(OPTIMIZERS))
        if opt_params['optimizer'] == 'adam':
            opt_params['optimizer'] = 1
        elif opt_params['optimizer'] == 'adagrad':
            opt_params['optimizer'] = 0
        elif opt_params['optimizer'] == 'sgd':
            opt_params['optimizer'] = 2

        self.nrow = None
        self.ncol = None
        self.K = K
        self.opt_params = opt_params

    def _init_params(self):
        DIV = 1000
        self.beta0 = np.random.normal(0, 1, 1) / DIV
        self.beta = np.random.normal(0, 1, self.ncol) / DIV
        self.v = np.random.normal(0, 1, (self.ncol, self.K)) / DIV

    def fit(self, Xvals, Xrows, Xcols, Yvals, weights=None):
        self.nrow = Xrows.max() + 1
        self.ncol = Xcols.max() + 1
        self._init_params()

        if weights is None:
            weights = np.repeat(1.0, self.nrow)

        self.coef_ = fit_fm(self.beta0, self.beta, self.v, self.opt_params, Xvals,
                            Xrows, Xcols, Yvals, self.nrow, self.ncol, weights)
        self._set_fitted_coef()
        return self

    def predict(self, Xvals, Xrows, Xcols):
        nrow = Xrows.max() + 1 
        return predictfm(self.f_beta0, self.f_beta, self.f_v, Xvals,
                         Xrows, Xcols, nrow, self.ncol)

    def loss(self, Xvals, Xrows, Xcols, Yvals):
        return sum((self.predict(Xvals, Xrows, Xcols) - Yvals)**2) / (Xrows.max() + 1)

    def _set_fitted_coef(self):
        self.f_beta0, self.f_beta, self.f_v = self.coef_
