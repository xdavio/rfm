import numpy as np

from c_fm import fit_fm, predictfm

OPTIMIZERS = ['adam', 'adagrad']

    # int minibatch; // minibatch count
    # int n_outer;   // maxiter
    # float eta;     // learning rate
    # float lambda;  // penalty on v
    # float eps;     // epsilon term for adam and adagrad
    # float beta1;   // adagrad-only
    # float beta2;   // adagrad-only


class FM:
    def __init__(self, nrow, ncol, K, opt_params):
        if opt_params['optimizer'] not in OPTIMIZERS:
            raise ValueError('Supplied optimizer is not one of ' +
                             ', '.join(OPTIMIZERS))
        if opt_params['optimizer'] == 'adam':
            opt_params['optimizer'] = 1
        elif opt_params['optimizer'] == 'adagrad':
            opt_params['optimizer'] = 0

        self.nrow = nrow
        self.ncol = ncol
        self.K = K
        self.opt_params = opt_params
        self._init_params()

    def _init_params(self):
        DIV = 1000
        self.beta0 = np.random.normal(0, 1, 1) / DIV
        self.beta = np.random.normal(0, 1, self.ncol) / DIV
        self.v = np.random.normal(0, 1, (self.ncol, self.K)) / DIV

    def fit(self, Xvals, Xrows, Xcols, Yvals, Yind):
        self.coef_ = fit_fm(self.beta0, self.beta, self.v, self.opt_params, Xvals,
                            Xrows, Xcols, Yvals, Yind, self.nrow, self.ncol)
        return self

    def predict(self, Xvals, Xrows, Xcols):
        return predictfm(self.beta0, self.beta, self.v, Xvals,
                         Xrows, Xcols, self.nrow, self.ncol)

    def loss(self, Xvals, Xrows, Xcols, Yvals):
        return sum((self.predict(Xvals, Xrows, Xcols) - Yvals)**2) / self.nrow
