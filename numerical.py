#! -*- coding:utf-8 -*-

from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


class PercentileBinner(BaseEstimator, TransformerMixin):
    """
        PercentileBinner transforms a continuous feature into a categorical one.

        First, given nb_bins, PercentileBinner computes the percentiles to use
        exactly nb_bins values in the output variable. Then, using numpy.digitize
        PercentileBinner will evenly discretize the variable


    Parameters:
    ----------
        nb_bins: integer, required
            The number of values the the output variable should take.

    Output:
    _______
        result: array-like  The digitized variables using percentile bins.
    """
    def __init__(self, nb_bins):
        self.nb_bins = nb_bins
        super(BaseEstimator, self).__init__()

    def fit(self, X, y=None):
        self.pc = np.percentile(X, np.linspace(100 / self.nb_bins, 100 - 100 / self.nb_bins, self.nb_bins))
        return self

    def transform(self, X):
        return np.digitize(X, bins=self.pc).reshape((len(X), 1))


class HistogramTransformer(BaseEstimator, TransformerMixin):
    """ This class wraps numpy.histogram and numpy.digitize in order to digitize a feature vector"""
    def init(self, bins=10, range=None, normed=False, weights=None, density=None, right=False):
        self.bins = bins
        self.range = range
        self.normed = normed
        self.weights = weights
        self.density = density
        self.right = right

    def fit(self, X, y=None):
        hist, bin_edges = np.histogram(X, bins=self.bins)
        self.bin_edges = bin_edges

    def transform(self, X, y=None):
        if self.bin_edges is None:
            raise
        return np.digitize(X, self.bins, right=self.right).reshape((len(X), 1))
