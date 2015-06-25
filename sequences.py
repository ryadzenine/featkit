#! -*- coding:utf-8 -*-

from sklearn.base import TransformerMixin


class SequenceEncodingTransformer(TransformerMixin):
    """ encodes a sequeces xyz as xy,yz,z"""
    def __init__(self, padding=1, overlap=False, backward=False):
        self.padding = padding
        self.overlap = overlap

    def fit(X, y=None):
        pass

    def transform(X, y=None):
        pass
