import os
import numpy as np
import typing
import pandas as pd

from base.proxy import Proxy, AntaresProxy
from tempo.trajectory import TempoTrajectory
from tempo.bellman import TempoBellman
from tempo.cost_function import TempoCostFunction
from tempo.reservoir import TempoReservoir
import constants


class TempoProxy(Proxy):
    """Proxy class for tempo. Inherits from Tempo

    Manages computation for tempo trajectories and controls upon a given set of scenario
    """
    data_first_month = 0

    def __init__(self, residual_load: np.ndarray[tuple[int, int], np.dtype[np.float64]],
                 reservoirs: typing.List[TempoReservoir],
                 data_first_month: int, day_first_january: int,
                 mc_years: np.ndarray | None = None,
                 ts_selection: np.ndarray | None = None,
                 leap_year: bool = False, c_var: float = 1.)\
            -> None:
        """Initialises the proxy

        Parameters:
            residual_load: Residual_load of the different scenarios provided (hourly).
            reservoirs: The reservoirs used for the simulation, first one must correspond to the
                tightest restrictions (red).
            data_first_month (int): First month of the data in int (0 for january, 11 for december).
            day_first_january (int): Day of the week for the first of january included in the data in int (0 for monday,
                6 for sunday)
            mc_years (np.ndarray(int)): List of years for which to compute trajectories.
            ts_selection (np.ndarray(int)): List of years to take into account to compute bellman values.
            leap_year (bool): Is the considered year a leap year?
            c_var (float): Keeps only the percentage worst case scenarii to calculate bellman values.
        """

        if leap_year:
            constants.MONTHS[1] = 29
        else:
            constants.MONTHS[1] = 28

        # computing first marsh day of the week
        if data_first_month == 1 or data_first_month == 2:
            day_first_marsh = (day_first_january + 2) % 7
        else:
            day_first_marsh = (day_first_january + 3) % 7

        # and then first september of previous year
        day_first_september = (day_first_marsh + 1) % 7

        for res in reservoirs:
            res.week_day_first_september = day_first_september

        # changing months order so that we start on september while keeping marsh days untouched
        self.data_first_month = data_first_month
        months = np.roll(constants.MONTHS, 4)
        self.roll_idx_month = 4 + self.data_first_month
        first_sept = months[self.roll_idx_month:].sum()
        first_sept = int(first_sept)
        # we remove last day so week_days are kept upon translation
        residual_load_364 = residual_load[:364, :]
        tempo_residual_load = residual_load_364

        self.roll_idx_day = 0
        if self.data_first_month <= 2 or self.data_first_month > 8:  # beginning is between september and marsh
            tempo_residual_load = np.concatenate((residual_load_364[first_sept-1:, :],
                                                 residual_load_364[:first_sept-1, :]))
            self.roll_idx_day = first_sept-1
        elif self.data_first_month == 3:
            residual_load_364 = residual_load[1:, :]
            tempo_residual_load = np.concatenate((residual_load_364[first_sept-1:, :],
                                                  residual_load_364[:first_sept-1, :]))
            self.roll_idx_day = first_sept-1
        elif self.data_first_month != 8:
            # on met le début à la fin
            tempo_residual_load = np.concatenate((residual_load_364[first_sept:, :],
                                                  residual_load_364[:first_sept, :]))
            self.roll_idx_day = first_sept

        super().__init__(tempo_residual_load, list(reservoirs), mc_years, ts_selection)
        nb_sce = self._residual_load.shape[1]
        for res in reservoirs:
            cost_function = TempoCostFunction(self._residual_load, res)
            self._cost_function.append(cost_function)
            bellman = TempoBellman(self.ts_selection, cost_function, res, c_var)
            self._bellman.append(bellman)
            prev_traj = None
            if len(self._trajectory):
                prev_traj = self._trajectory[-1]
            trajectories = TempoTrajectory(self.mc_years, res, cost_function, bellman, prev_traj)
            self._trajectory.append(trajectories)

    def _roll_back_day(self, array: np.ndarray) -> np.ndarray:
        """Puts a daily array back in the order of input data (first begining month)."""
        return np.concatenate((array[:, -self.roll_idx_day:], array[:, :-self.roll_idx_day]), axis=1)

    def _roll_back_week(self, array: np.ndarray) -> np.ndarray:
        """Puts a weekly array back in the order of input data (first begining month)."""
        # idx from 1st january
        first_monday_september = constants.MONTHS[:8].sum()
        res = self._reservoir[0]
        assert isinstance(res, TempoReservoir)
        if res.week_day_first_september != 0:
            first_monday_september += 7 - res.week_day_first_september
        first_day_res = constants.MONTHS[:self.data_first_month].sum()
        nb_weeks = (first_monday_september - first_day_res) // 7
        if nb_weeks < 0:
            nb_weeks += 1
        return np.concatenate((array[:, -nb_weeks:], array[:, :-nb_weeks]), axis=1)

    def get_trajectories(self) -> typing.List[np.ndarray]:
        """Returns the computed trajectories."""
        res: list[np.ndarray] = []
        for t in self._trajectory:
            res.append(self._roll_back_week(t.get_trajectories()))
        return res

    def get_daily_controls(self) -> np.ndarray:
        """
        Returns daily control trajectories (red and white) for all scenarios and weeks.
        The daily net loads are sorted and matched to the controls.
        """
        red_controls = self._trajectory[0].get_controls()
        white_controls = self._trajectory[1].get_controls() - red_controls
        nb_scenarios = red_controls.shape[0]
        daily_trajectory = np.asarray([["bleu "] * (constants.NB_DAYS + 1)] * nb_scenarios)
        net_load = self._residual_load[:, self.ts_selection]
        res = self._reservoir[0]
        assert isinstance(res, TempoReservoir)

        first_monday_september = 0
        if res.week_day_first_september != 0:
            first_monday_september += 7 - res.week_day_first_september

        for s in range(nb_scenarios):
            control_r = red_controls[s]
            control_w = white_controls[s]

            for week in range(constants.RESULTS_SIZE):
                week_start = week * 7 + first_monday_september
                week_end = week_start + 7
                week_days = net_load[week_start: week_end, s]
                week_days_r = week_days[:5]
                week_days_w = week_days[:6]

                sorted_days_r = np.argsort(week_days_r)[::-1]
                sorted_days_w = np.argsort(week_days_w)[::-1]

                r = int(control_r[week]) if control_r[week] is not None else None
                w = int(control_w[week]) if control_w[week] is not None else None

                used_days = set()

                for d in sorted_days_r:
                    if r is not None and r > 0:
                        used_days.add(d)
                        r -= 1
                        day = week_start + d
                        if day >= constants.NB_DAYS:
                            day -= constants.NB_DAYS
                        daily_trajectory[s][day] = "rouge"

                for d in sorted_days_w:
                    if w is not None and w > 0 and d not in used_days:
                        used_days.add(d)
                        w -= 1
                        day = week_start + d
                        if day >= constants.NB_DAYS:
                            day -= constants.NB_DAYS
                        daily_trajectory[s][day] = "blanc"
        daily_trajectory = self._roll_back_day(np.asarray(daily_trajectory))
        return daily_trajectory

    def get_controls(self) -> typing.List[np.ndarray]:
        """Returns the computed controls."""
        res: list[np.ndarray] = []
        for t in self._trajectory:
            res.append(self._roll_back_week(t.get_controls()))
        return res

    def get_usage_values(self) -> typing.List[np.ndarray]:
        """Returns the computed usage values."""
        res: list[np.ndarray] = []
        for b in self._bellman:
            res.append(self._roll_back_week(b.get_usage_values()))
        return res

    def get_bellman_values(self) -> typing.List[np.ndarray]:
        """Returns the computed bellman values."""
        res: list[np.ndarray] = []
        for b in self._bellman:
            res.append(self._roll_back_week(b.get_bellman_values()))
        return res


class TempoAntaresProxy(AntaresProxy):
    """This class manages the computation of Bellman values and trajectory regarding a tempo reservoir
    using the scenarii of a given antares study."""
    def __init__(self, study_path: str, area_name: str, mc_years: np.ndarray, sce_selection: np.ndarray | None = None, c_var: float = 1.):
        super().__init__(study_path, area_name, mc_years, sce_selection)
        weekday_1_jan = AntaresProxy._int_from_antares_weekday(
            self.study.get_settings().general_parameters.january_first
        )
        first_month = AntaresProxy._int_from_antares_month(
            self.study.get_settings().general_parameters.first_month_in_year
        )
        reservoir_red = TempoReservoir(capacity=22,
                                       initial_level=22,
                                       excluded_week_days=np.asarray([5, 6]),
                                       first_day=61,
                                       last_day=211
                                       )
        reservoir_white = TempoReservoir(capacity=65,
                                         initial_level=65,
                                         excluded_week_days=np.asarray([6]),
                                         first_day=0,
                                         last_day=constants.NB_DAYS-1
                                         )
        self._residual_load = self._area_loads[area_name]
        self._residual_load = self._residual_load.reshape(
            (constants.NB_DAYS + 1, 24, self._residual_load.shape[1])
        ).sum(axis=1)

        self._proxy = TempoProxy(self._residual_load, [reservoir_red, reservoir_white],
                                 first_month, weekday_1_jan,
                                 mc_years, sce_selection,
                                 self.study.get_settings().general_parameters.leap_year, c_var)

    def export_controls(self, export_dir: str, filename: str = "controls.csv") -> None:
        """
        Export optimal control trajectories
        for all scenarios and weeks to a CSV file.
        """
        controls = self._proxy.get_controls()
        controls_red = controls[0]
        controls_white = controls[1] - controls[0]
        data = []
        for sce_ind in range(controls_red.shape[0]):
            for week_ind in range(controls_red.shape[1]):
                data.append({
                    "area": self.area,
                    "red_days_control": controls_red[sce_ind, week_ind],
                    "white_days_control": controls_white[sce_ind, week_ind],
                    "week": week_ind + 1,
                    "mcYear": sce_ind + 1
                })

        df = pd.DataFrame(data)
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        output_path = os.path.join(export_dir, filename)
        df.to_csv(output_path, index=False)

    def export_trajectories(self, export_dir: str, filename: str = "trajectories.csv") -> None:
        """
        Export optimal stock trajectories for all scenarios and weeks
        to a CSV file.
        """
        trajectories = self._proxy.get_trajectories()
        trajectories_red = trajectories[0]
        trajectories_white = trajectories[1] - trajectories[0]
        data = []
        for sce_ind in range(trajectories_red.shape[0]):
            for week_ind in range(trajectories_red.shape[1]):
                data.append({
                    "area": self.area,
                    "red_days_remaining": trajectories_red[sce_ind, week_ind],
                    "white_days_remaining": trajectories_white[sce_ind, week_ind],
                    "week": week_ind + 1,
                    "mcYear": sce_ind + 1
                })
        df = pd.DataFrame(data)
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        output_path = os.path.join(export_dir, filename)
        df.to_csv(output_path, index=False)

    def export_daily_controls(self, sce: int, export_dir: str, filename: str = "") -> None:
        """
        Export optimal daily control trajectories
        for all scenarios and weeks to a CSV file.
        """
        assert isinstance(self._proxy, TempoProxy)
        controls = self._proxy.get_daily_controls()
        df = pd.DataFrame(controls[sce])
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        if filename == "":
            filename = f"daily_controls_{sce}.csv"
        output_path = os.path.join(export_dir, filename)
        df.to_csv(output_path, index=False)
