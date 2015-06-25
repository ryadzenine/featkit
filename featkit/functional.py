#! -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class FuncTransformer(BaseEstimator, TransformerMixin):
    """ This transformer applies the provided function to the feature vector

    Parameters
    ----------
    func: function, required
        The function to be applied to each observation of the feature
    lazy: boolean, optionnal
        Whether functionTransformer returns a generator (lazy = True) or a list
    """
    def __init__(self, func=None):
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.func(X).reshape((len(X), 1))


class PandasFuncTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, func=None):
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.func(X).reshape((len(X), 1))


class LineFuncTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, func=None):
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([self.func(x) for x in X]).reshape((len(X), 1))
