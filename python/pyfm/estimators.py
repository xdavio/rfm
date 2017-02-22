import numpy as np

from c_fm import fit_fm, predictfm

OPTIMIZERS = ['adam', 'adagrad']


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
        self._set_fitted_coef()
        return self

    def predict(self, Xvals, Xrows, Xcols, nrow, ncol):
        return predictfm(self.f_beta0, self.f_beta, self.f_v, Xvals,
                         Xrows, Xcols, nrow, ncol)

    def loss(self, Xvals, Xrows, Xcols, Yvals, nrow, ncol):
        return sum((self.predict(Xvals, Xrows, Xcols, nrow, ncol) - Yvals)**2) / nrow

    def _set_fitted_coef(self):
        self.f_beta0, self.f_beta, self.f_v = self.coef_


class FMEpoch(FM):
    def __init__(self, nrow, ncol, K, opt_params):
        self.n_epochs = opt_params.get('n_epochs', 100)
        self.epoch_loss = np.zeros(self.n_epochs)

        super().__init__(nrow, ncol, K, opt_params)

    def _fit(self, Xvals, Xrows, Xcols, Yvals, Yind):
        s = super().fit(Xvals, Xrows, Xcols, Yvals, Yind)
        self.beta0, self.beta, self.v = s.coef_

    def fit(self, Xvals, Xrows, Xcols, Yvals, Yind):
        """
        BUG: requires that the sparse representation be rebuilt for each epoch
        """
        for i in range(self.n_epochs):
            self._fit(Xvals, Xrows, Xcols, Yvals, Yind)
            self.epoch_loss[i] = self.loss(Xvals, Xrows, Xcols, Yvals,
                                           self.nrow, self.ncol)
        return self
