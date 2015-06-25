# coding: utf-8

from sklearn.base import TransformerMixin, BaseEstimator


class DebugSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=0):
        try:
            print X.shape
        except:
            pass

        return self

    def transform(self, X):
        return X
