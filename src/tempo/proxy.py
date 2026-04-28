import os
import numpy as np
import typing
import pandas as pd

from proxy import Proxy, AntaresProxy
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
                 reservoirs: typing.List[TempoReservoir], data_first_month, day_first_january, c_var: float = 1.)\
            -> None:
        """Initialises the proxy

        Parameters:
            residual_load: residual_load of the different scenarios provided (hourly)
            reservoirs: the reservoirs used for the simulation, first one must correspond to the
                tightest restrictions (red)
            c_var (float): keep only the percentage worst case scenarii to calculate bellman values
        """

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
        first_sept = months[4 + self.data_first_month:].sum()
        first_sept = int(first_sept)
        # we remove last day so week_days are kept upon translation
        residual_load_364 = residual_load[:364, :]
        tempo_residual_load = residual_load_364
        if self.data_first_month <= 2 or self.data_first_month > 8:  # beginning is between september and marsh
            tempo_residual_load = np.concatenate((residual_load_364[first_sept-1:, :],
                                                 residual_load_364[:first_sept-1, :]))
        elif self.data_first_month == 3:
            residual_load_364 = residual_load[1:, :]
            tempo_residual_load = np.concatenate((residual_load_364[first_sept-1:, :],
                                                  residual_load_364[:first_sept-1, :]))
        elif self.data_first_month != 8:
            # on met le début à la fin
            tempo_residual_load = np.concatenate((residual_load_364[first_sept:, :],
                                                  residual_load_364[:first_sept, :]))

        super().__init__(tempo_residual_load, list(reservoirs))
        nb_sce = self._residual_load.shape[1]
        for res in reservoirs:
            cost_function = TempoCostFunction(self._residual_load, res)
            self._cost_function.append(cost_function)
            bellman = TempoBellman(nb_sce, cost_function, res, c_var)
            self._bellman.append(bellman)
            prev_traj = None
            if len(self._trajectory):
                prev_traj = self._trajectory[-1]
            trajectories = TempoTrajectory(nb_sce, res, cost_function, bellman, prev_traj)
            self._trajectory.append(trajectories)


class TempoAntaresProxy(AntaresProxy):
    def __init__(self, study_path: str, area_name: str, mc_years: int, sce_selection: list[int] | None = None, c_var: float = 1.):
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
                                 first_month, weekday_1_jan, c_var)

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
        output_path = os.path.join(export_dir, filename)
        df.to_csv(output_path, index=False)
