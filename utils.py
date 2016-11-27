#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 05 17:52:25 2014

@author: han.yan
"""

from sys import platform
from datetime import datetime, timedelta
from calendar import monthrange
import numpy as np
from numba import jit
from time import mktime
import nanotime

# import warnings
# from pandas.tseries.holiday import USFederalHolidayCalendar
# from pandas.tseries.offsets import CustomBusinessDay as CBD
# from pandas.io.pytables import PerformanceWarning

# from ciso8601 import parse_datetime
# platform specific


miu = 'u' if platform == 'win32' else unichr(0x3bc).encode('utf-8')


def to_timestamp(x):
    return (x - datetime(1970, 1, 1)).total_seconds()


def timeflag():
    if platform == 'win32':
        return to_utctimestamp(datetime.utcnow())
    else:
        return nanotime.now().timestamp()


def timing(t0=None, string=None, unit='s'):
    fold = {'min': 1. / 60, 's': 1, 'ms': 1000, 'us': 1e6, 'ns': 1e9}
    if t0 is None:
        return timeflag() * fold[unit]

    new_unit = unit.replace('us', miu + 's')
    print ' <-- %s = %.2f %s' % (
        string, (timeflag() - t0) * fold[unit], new_unit)
    return timeflag()


def to_utctimestamp(dt):
    utc_naive = dt.replace(tzinfo=None)
    if dt.utcoffset():
        utc_naive -= dt.utcoffset()
    return (utc_naive - datetime(1970, 1, 1)).total_seconds()


# def today_datetime():
#     return datetime.combine(datetime.today().date(), datetime.min.time())


def timediff(time1, time2, target='min'):
    t1 = time1.hour * 60.0 + time1.minute + \
        time1.second / 60.0 + time1.microsecond / 60e6
    t2 = time2.hour * 60.0 + time2.minute + \
        time2.second / 60.0 + time1.microsecond / 60e6
    return t1 - t2


def yyyymm_to_datetime(index):
    index = [datetime.strptime(str(x), '%Y%m') +
             timedelta(days=40) for x in index]
    return [x.replace(day=1) + timedelta(days=-1) for x in index]


@jit(nopython=True)
def locate_dates_jit(search, target, round_down):
    res = -np.ones(len(search))
    loc = res.copy()

    for i in range(len(search)):
        x = search[i]
        best = -1
        index = -1
        for j in range(len(target)):
            y = target[j]
            if y == x:
                best = x
                index = j
                break
            if y > x and not round_down and (index < 0 or y < best) or \
                    y < x and round_down and (index < 0 or y > best):
                best = y
                index = j

        res[i] = best
        loc[i] = index

    return res, loc


def locate_dates(search, target, round_down=False, intraday=False):
    """locating dates.
    """
    if not hasattr(search, '__iter__'):
        search = [search]
    if not hasattr(target, '__iter__'):
        raise Exception('target must be iterable...')

    if intraday:
        search = [mktime(x.utctimetuple()) for x in search]
        target = [mktime(x.utctimetuple()) for x in target]
    else:
        search = [x.toordinal() for x in search]
        target = [x.toordinal() for x in target]

    # jit routine for fast matching
    search = np.array(search)
    target = np.array(target)
    res, loc = locate_dates_jit(search, target, round_down)

    if intraday:
        res = [datetime.utcfromtimestamp(
            x) if x is not None else None for x in res]
    else:
        res = [datetime.fromordinal(int(x)) if x > 0 else None for x in res]
        res = np.array(res)

    return res, loc.astype(int)


def timeseq(start=None, end=None, steps=10, step_min=1):
    seq = []
    if end is not None and start is None:
        seq = [end + timedelta(minutes=-x * step_min) for x in range(0, steps)]
        seq.reverse()
    elif end is not None and start is not None:
        cursor = start
        while cursor <= end:
            seq.append(cursor)
            cursor = cursor + timedelta(minutes=step_min)
    else:
        raise Exception('function not implemented yet...')

    return seq


def end_of_month(start_datetime, end_datetime=datetime.utcnow()):
    start_year = start_datetime.year
    end_year = end_datetime.year

    for i in np.arange(start_year, end_year + 1):
        for j in range(12):
            if i == start_year and j + 1 < start_datetime.month or \
               i == end_year and j + 1 > end_datetime.month:
                continue
            day = monthrange(i, j + 1)[1]
            date = datetime(i, j + 1, day).date()
            if date <= end_datetime.date():
                yield date


def to_month_end(dates):
    for y in dates:
        yield y.replace(day=monthrange(y.year, y.month)[1])


if __name__ == '__main__':

    x = np.random.random(25).reshape(5, 5)
    y = np.random.random(25).reshape(5, 5)
