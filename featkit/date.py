#! -*- coding:utf-8 -*-

from sklearn.base import TransformerMixin
import numpy as np

from datetime import datetime


def _date_time_parser(date, date_format="timestamp"):
    if date_format == "timestamp":
        return datetime.fromtimestamp(date)
    else:
        return datetime.strptime(date, date_format)


class _BaseDateTransformer(TransformerMixin):
    def __init__(self, date_format="timestamp"):
        self.fmt = date_format

    def fit(self, X, y=None):
        """ No op """
        return self


class DayOfTheWeek(_BaseDateTransformer):
    """ Extracts the day of the week """
    def _day_of_the_week(self, date):
        return _date_time_parser(date, self.fmt).isoweekday()

    def transform(self, X, y=None):
        d = np.vectorize(self._day_of_the_week)
        return d(X).reshape(len(X), 1)


class IsWeekend(DayOfTheWeek):
    def _is_week_end(self, date):
        return self._day_of_the_week(date) in [6, 7]

    def transform(self, X, y=None):
        d = np.vectorize(self._is_week_end)
        return d(X).reshape(len(X), 1)


class DayOfTheMonth(_BaseDateTransformer):
    def _day_of_the_month(self, date):
        return _date_time_parser(date, self.fmt).day

    def transform(self, X, y=None):
        d = np.vectorize(self._day_of_the_month)
        return d(X).reshape(len(X), 1)


class DayOfTheYear(_BaseDateTransformer):
    def _day_of_the_year(self, date):
        return _date_time_parser(date, self.fmt).timetuple()[7]

    def transform(self, X, y=None):
        d = np.vectorize(self._day_of_the_year)
        return d(X).reshape(len(X), 1)


class MonthOfTheYear(_BaseDateTransformer):
    def _month_of_the_year(self, date):
        return _date_time_parser(date, self.fmt).month

    def transform(self, X, y=None):
        d = np.vectorize(self._month_of_the_year)
        return d(X).reshape(len(X), 1)


class Seasons(_BaseDateTransformer):
    def _season_of_the_year(self, date):
        return (_date_time_parser(date, self.fmt).month - 1) / 3

    def transform(self, X, y=None):
        d = np.vectorize(self._season_of_the_year)
        return d(X).reshape(len(X), 1)


class DayOrNight(_BaseDateTransformer):
    def __init__(self, hours_mapping=None, date_format="timestamp"):
        self.hours_mapping = hours_mapping
        super(_BaseDateTransformer, self).__init__(date_format)

    def _period_of_the_day(self, date):
        return _date_time_parser(date, self.fmt).hour < 20 and _date_time_parser(date, self.fmt).hour > 6

    def transform(self, X, y=None):
        d = np.vectorize(self._period_of_the_day)
        return d(X).reshape(len(X), 1)
