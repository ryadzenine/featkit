#! -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import label_binarize


class ThresholdLabelBinarizer(BaseEstimator, TransformerMixin):
    """ ThresholdLabelBinarizer is typically used with categorical features that takes too many
    values. The idea here is to only consider nb_bins values, Then encode everything else
    with the replacement_label provided.

    Example:
    -------
        >>> ThresholdLabelBinarizer(nb_bins=2, binarize=False, replacement_label=3).fit_transform([1,2,2,1,5,6])
        np.array([1, 2, 2, 1, 3, 3])
    """
    def __init__(self, nb_bins=10, binarize=True, replacement_label=None):
        if replacement_label is None:
            raise Exception("replacement_label is mandatory, should not take None Value")
        self.replacement = replacement_label
        self.bins = nb_bins
        self.binarize = binarize

    def _replace_label(self, x):
        if x not in self.labels:
            return self.replacement
        return x

    @staticmethod
    def other_label(serie, nb_classes, replacement_class):
        """ keeps the most used classes and encodes the rest as replacement_class """
        classes = serie.value_counts()[:nb_classes-1].keys()

        def fn(x):
            if x in classes:
                return x
            return replacement_class
        return (pd.Series([fn(x) for x in serie]), classes)

    def fit(self, X, y=None):
        X_t, self.labels = ThresholdLabelBinarizer.other_label(X, self.bins, self.replacement)
        return self

    def transform(self, X, y=None):
        f = np.vectorize(self._replace_label)
        X_t = f(X).reshape(len(X), 1)
        if self.binarize:
            return label_binarize(X_t, classes=self.labels)
        else:
            return X_t
