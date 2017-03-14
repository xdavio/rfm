import numpy as np
from sklearn.model_selection import KFold

from c_fm import fit_fm, predictfm

OPTIMIZERS = ['adam', 'adagrad', 'sgd']
LOSS = ['linear', 'binary', 'cross_entropy']


class FM:
    def __init__(self, K, opt_params, standardize=False):
        opt_params = opt_params.copy()
        
        if opt_params['optimizer'] not in OPTIMIZERS:
            raise ValueError('Supplied optimizer is not one of ' +
                             ', '.join(OPTIMIZERS))
        if opt_params['optimizer'] == 'adam':
            opt_params['optimizer'] = 1
        elif opt_params['optimizer'] == 'adagrad':
            opt_params['optimizer'] = 0
        elif opt_params['optimizer'] == 'sgd':
            opt_params['optimizer'] = 2

        # get response type
        loss = opt_params.get('loss', 'linear')        
        if loss not in LOSS:
            raise ValueError('Supplied response type is not one of ' +
                             ', '.join(OPTIMIZERS))
        if loss == 'linear':
            opt_params['loss'] = 0
        elif loss == 'binary':
            opt_params['loss'] = 1
        elif loss == 'cross_entropy':
            opt_params['loss'] = 2

        self.nrow = None
        self.ncol = None
        self.K = K
        self.opt_params = opt_params
        self.standardize = standardize
        self.y_mean = None
        self.y_std = None

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

        if self.standardize:
            self.y_mean = Yvals.mean()
            self.y_std = Yvals.std()
            Yvals = (Yvals - self.y_mean) / self.y_std

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

        if self.standardize:
            self.f_beta0 = self.y_mean + self.y_std * self.f_beta0
            self.f_beta *= self.y_std
            self.f_v *= np.sqrt(self.y_std)


class FMCV:
    def __init__(self, K, opt_params, standardize=False,
                 n_splits=5):
        self.n_splits = n_splits
        self.cv = KFold(n_splits)

        self.fm = FM(K, opt_params, standardize)
        
    def fit(self, Xvals, Xrows, Xcols, Yvals, weights=None):
        if weights is None:
            weights = np.repeat(1.0, self.fm.nrow)

        preds = []
        tests = []
        for train, test in self.cv.split(Xvals):
            # this is wrong, have to handle rows w.r.t. Xvals
            self.fm.fit(Xvals[train], Xrows[train], Xcols[train],
                        Yvals[train], weights=weights[train])
            preds.append(self.fm.predict(Xvals[test], Xrows[test], Xcols[test]))
            tests.append(test)
        self.preds = np.concatenate(tuple(preds))
        self.tests = np.concatenate(tuple(tests))

        self.resids = self.preds - Yvals[self.tests]
        self.cv_loss = np.mean(self.resids ** 2)
