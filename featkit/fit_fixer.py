# coding: utf-8

from sklearn.base import TransformerMixin, BaseEstimator

class FitFixer(BaseEstimator, TransformerMixin):
    def __init__(self, cls):
        self.cl = cls

    def fit(self, X, y=None):
        self.cl.fit(X)
        return self

    def transform(self, X):
        return self.cl.transform(X)
