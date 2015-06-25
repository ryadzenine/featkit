#! -*- coding: utf-8 -*-


import unittest
from categorical import ThresholdLabelBinarizer
from pandas import Series


class TestThresholdLabelBinarizer(unittest.TestCase):
    def test_other_label(self):
        tests = [
            {"serie": ["1", "2", "2", "0", "0", "0"],
             "replace": "3",
             "nb_cls": 3,
             "result": ["1", "2", "2", "0", "0", "0"]},
            {"serie": ["1", "2", "2", "0", "0", "0"],
             "replace": "3",
             "nb_cls": 2,
             "result": ["3", "2", "2", "0", "0", "0"]},
            {"serie": [1, 2, 2, 0, 0, 0],
             "replace": 3,
             "nb_cls": 2,
             "result": [3, 2, 2, 0, 0, 0]}]
        for case in tests:
            self.assertTrue(
                ThresholdLabelBinarizer.other_label(Series(case["serie"]), case["nb_cls"], case["replace"]),
                Series(case["result"]))
