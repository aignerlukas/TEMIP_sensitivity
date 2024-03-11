#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:03:43 2021

timer class from:
    https://realpython.com/python-timer/

updated with formatting to hh:mm:ss.sssss

@author: laigner
"""



# timer.py


import time
import datetime


class TimerError(Exception):

    """A custom exception used to report errors in use of Timer class"""


class Timer:

    def __init__(self):
        self._start_time = None
        self._stop_time = None


    def start(self):
        """Start a new timer"""

        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()


    def stop(self, prefix='run-'):
        """Stop the timer, and report the elapsed time"""

        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        self._stop_time = time.perf_counter()
        elapsed_time = self._stop_time - self._start_time
        elapsed_time_string = str(datetime.timedelta(seconds=elapsed_time))[:-3]
        self._start_time = None

        print(f"Elapsed {prefix}time (hh:mm:ss.sss): {elapsed_time_string:s}")

        return elapsed_time, elapsed_time_string