"""
This module defines the constants that are relevant for all proxies
"""

import numpy as np


NB_HOURS: int = 8760
"""Number of hours in a year."""
NB_DAYS: int = 364
"""Number of days in a year."""
NB_HOURS_IN_DAY = 24
"""Number of hours in a day."""
RESULTS_SIZE: int = 52
"""Expected size of the results. Currently equal to the number of weeks in a year."""
RESULTS_INTERVAL_DAYS: int = 7
"""Expected interval for each value of the results in day, currently a week."""
RESULTS_INTERVAL_HOURS: int = 7*24
"""Expected interval for each value of the results in hours, currently a week."""
MONTHS: np.ndarray = np.asarray([31,  # january
                                 28,    # february
                                 31,    # march
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
"""Number of days in each month of a non-leap year."""
