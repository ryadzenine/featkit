import unittest

from datetime import datetime
from calendar import timegm
from .date import _date_time_parser

class TestDates(unittest.TestCase):
    def test_date_time_parser(self):
        dt = datetime.now()
        ti = timegm(dt.timetuple())
        self.assertTrue(_date_time_parser(ti, "timestamp"), dt)

    def test_day_of_the_week_(self):
        pass
    def test_day_of_the_month_(self):
        pass
    def test_day_of_the_year_transfomrer(self):
        pass
    def test_month_of_the_year_(self):
        pass
    def test_seansons_tranformer(self):
        pass
    def test_period_of_the_day(self):
        pass
