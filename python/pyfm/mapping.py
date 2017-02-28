import numpy as np
import pandas as pd

"""
this file attempts to build out logic for keeping track of 
features embedded in a sparse space
"""

def _mapping_to_triplet(mapping):
    values = np.concatenate(tuple(mapping[var]['value'] for var in mapping))
    rows = np.concatenate(tuple(mapping[var]['row'] for var in mapping))
    cols = np.concatenate(tuple(mapping[var]['col'] for var in mapping))
    return values, rows, cols

class SparseEmbedding:
    def __init__(self):
        pass

    def fit_pd(self, df):
        """
        df: pd.DataFrame
        """
        self.fit(df.values, df.columns)
    
    def fit(self, X, columns):
        """
        Fits the sparse embedding into values, rows, columns.

        `columns` identifies the different variables in the embedding.
        Should be a list of hashable objects (e.g. list of str)        
        """
        self.columns = columns
        self.nrow, self.ncol = X.shape

        if len(columns) != self.ncol:
            raise Exception("ncol not equal to number of column names")

        # mapping dict ( var : dict ( id2code, code2id, max_id, vals, rows, cols )
        mapping = {}
        last_max = 0
        for i, var in enumerate(columns):
            # unique values
            unique_values = np.unique(X[:, i])
            no_values = unique_values.shape[0]

            if no_values <= 1:
                # don't include a constant variable
                continue
            # TODO: make this commented block work correctly
            # elif no_values == 2:
            #     # if there are just 2 values, only embed it in a single column
            #     unique_values = np.array(unique_values[-1]).reshape(-1)
            #     no_values = 1

            mapping[var] = self._embed_vec(unique_values, no_values,
                                           X[:, i], last_max)
            
            # shift IDs by no_values in previous feature
            last_max += no_values

        self.mapping = mapping
        self.last_max = last_max

        return self

    def add_feature(self, var, datavec):
        unique_values = np.unique(datavec)
        no_values = unique_values.shape[0]
        self.mapping[var] = self._embed_vec(unique_values, no_values,
                                            datavec, self.last_max)
        self.last_max += no_values

    def _embed_vec(self, unique_values, no_values, datavec, last_max):
        m = {}
        m['codes'] = unique_values
        m['no_values'] = no_values
        m['ids'] = np.arange(last_max, last_max + no_values)

        id2code = {i:code for i, code in zip(m['ids'],
                                             m['codes'])}
        m['id2code'] = id2code
        code2id = {code:i for i, code in zip(m['ids'],
                                             m['codes'])}
        m['code2id'] = code2id

        m['row'] = np.arange(self.nrow)
        m['col'] = np.array([code2id[code] for code in datavec])
        m['value'] = np.repeat(1, self.nrow)

        return m
    
    def get_triplets(self):
        """
        Returns the triplets
        """
        self.values, self.rows, self.cols = _mapping_to_triplet(self.mapping)
        return self.values, self.rows, self.cols

    def _var_code2id(self, var, code):
        return self.mapping[var]['code2id'][code]

    def transform(self, X, columns):
        mapping = {}
        for i, var in enumerate(self.columns):
            if var not in columns:
                continue
            mapping[var] = {}
            mapping[var]['row'] = np.arange(X.shape[0])
            mapping[var]['col'] = np.array(
                [self._var_code2id(code) for code in X[:, i]])
            mapping[var]['value'] = np.repeat(1, X.shape[0])
        
        return _mapping_to_triplet(mapping)

    def list_vars(self):
        return [k for k in self.mapping]

    def var_agg(self, var):
        """
        returns {code: np.array of rows}
        """
        m = self.mapping[var]
        agg = {}

        for code in m['codes']:
            agg[code] = np.where(m['col'] == m['code2id'][code])[0]

        return agg

    def get_ids(self, var):
        return self.mapping[var]['ids']

    def get_codes(self, var):
        return self.mapping[var]['codes']

    def make_code_coef_df(self, var, coef, code_translate=None, sort=True):
        """
        `coef` 
        """
        df = pd.DataFrame({
            'code': self.get_codes(var),
            'coef': coef[self.get_ids(var)]
        })
        if code_translate is not None:
            df['name'] = df['code'].apply(code_translate)
        if sort:
            df = df.sort_values('coef', ascending=False)
        return df
        
    def make_interaction_df(self, var1, var2, coef_main, coef_int, sort=True,
                            code1_translate=None, code2_translate=None):
        code1 = self.get_codes(var1)
        ids1 = self.get_ids(var1)
        coef1 = coef_main[ids1]
        left = pd.DataFrame({'code': code1, 'id': ids1, 'coef': coef1})
        left['tmp'] = 1

        code2 = self.get_codes(var2)
        ids2 = self.get_ids(var2)
        coef2 = coef_main[ids2]
        right = pd.DataFrame({'code': code2, 'id': ids2, 'coef': coef2})
        right['tmp'] = 1

        # cartesian product
        m = left.merge(right, 'outer', on='tmp', suffixes=['_1', '_2'])
        m = m.drop('tmp', axis=1)

        m['interaction'] = m[['id_1', 'id_2']].apply(lambda x: (coef_int[x[0],:] *\
                                                                coef_int[x[1],:]).sum(),
                                                     axis=1)

        if sort:
            m = m.sort_values('interaction', ascending=False)

        if code1_translate is not None:
            m['name_1'] = m['code_1'].apply(code1_translate)
        if code2_translate is not None:
            m['name_2'] = m['code_2'].apply(code2_translate)

        return m

