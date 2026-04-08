import pytest
import numpy as np

from hydro.bellman import HydroBellman
from hydro.cost_function import HydroCostFunction
from hydro.reservoir import HydroReservoir
import constants


def test_zero_net_load():
    nb_sce = 1
    max_weekly_pump = np.ones(shape=constants.RESULTS_SIZE, dtype=np.float64)*10
    max_hourly_pump = np.ones(shape=constants.NB_HOURS, dtype=np.float64)*10 / (7 * 24)
    reservoir = HydroReservoir(initial_level=50,
                               final_level=50,
                               hourly_max_pump=max_hourly_pump,
                               hourly_max_turb=max_hourly_pump,
                               weekly_max_pump=max_weekly_pump,
                               weekly_max_turb=max_weekly_pump
                               )
    net_load = np.zeros(shape=(constants.NB_HOURS, nb_sce), dtype=np.float64)
    cost_function = HydroCostFunction(net_load, reservoir)
    bellman = HydroBellman(nb_sce, 1., cost_function, reservoir)
    values = bellman.get_bellman_values()
    assert isinstance(values, np.ndarray)
    for i in range(constants.RESULTS_SIZE):
        assert values[i, 50] == 0
        assert values[i, 49] > 0
        assert values[i, 51] > 0
