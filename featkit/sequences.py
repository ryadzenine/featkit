#! -*- coding:utf-8 -*-

from sklearn.base import TransformerMixin
from sklearn.feature_extraction import FeatureHasher


class SequenceHasher(TransformerMixin):
    """ encodes a sequeces xyz as xy,yz,z"""
    def __init__(self, base_feature_name="index"):
        self.base_feature_name = base_feature_name
        self.hasher = FeatureHasher(input_type="pair")

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        f_name = self.base_feature_name
        seq = (((f_name + str(i), v) for i, v in enumerate(x)) for x in X)
        return self.hasher.transform(seq)
