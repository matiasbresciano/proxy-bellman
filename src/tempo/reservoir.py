from dataclasses import dataclass, field
import numpy as np

from reservoir import Reservoir
import constants


@dataclass
class TempoReservoir(Reservoir):
    """Reservoir class for tempo. Inherits from Reservoir

    Attributes:
        excluded_week_days: days of the week when a tempo day cannot be used
        first_day: first day of the year when a tempo day can be used (included) (1st of november for red)
        last_day: last day of the year when a tempo day can be used (included) (31st of marsh for red)
    """
    excluded_week_days: np.ndarray = field(default_factory=
                                           lambda: np.asarray([5, 6], dtype=np.int16))
    first_day: int = 61  # november 1st
    last_day: int = 211  # march 31st
    week_day_first_september: int = field(default=0, init=False)  # september 1st is a monday

    def day_of_year_from_september(self, day: int, month: int) -> tuple[int, int]:
        """Returns the day of the year and day of the week associated to a day of a month

        1st September is 0, 31st August is 364
        parameters:
            day (int): day of the month, from 0 to month size - 1
            month (int): month with 0 for january, 11 for december
        """
        months = np.roll(constants.MONTHS, 4)
        day_of_year = months[0: (month-8) % 12].sum() + day
        day_of_week = (day_of_year + self.week_day_first_september) % 7
        return day_of_year, day_of_week

    def get_previous_monday(self, day_of_year: int) -> int:
        """Returns the previous monday

        If the given day is a monday, returns the same day
        parameters:
            day_of_year (int): day of the year, 1st September is 0, 31st August is 364
        """
        day_of_week = (day_of_year + self.week_day_first_september) % 7
        return (day_of_year - day_of_week) % (constants.NB_DAYS + 1)

