import numpy as np

"""
Constants for all proxies
"""
NB_HOURS: int = 8760  # nb hours in a year
NB_DAYS: int = 364  # nb days in a year
NB_HOURS_IN_DAY = 24
RESULTS_SIZE: int = 52  # nb weeks in a year
RESULTS_INTERVAL_DAYS: int = 7  # one week
RESULTS_INTERVAL_HOURS: int = 7*24  # one week in hours
MONTHS: np.ndarray = np.asarray([31,  # january
                                 28,    # february
                                 31,    # marsh
                                 30,    # april
                                 31,    # may
                                 30,    # june
                                 31,    # july
                                 31,    # august
                                 30,    # september
                                 31,    # october
                                 30,    # november
                                 31     # december
                                 ], dtype=np.int16)
