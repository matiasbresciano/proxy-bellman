import pytest
import numpy as np

from hydro.hydro_gain_function import HydroGainFunction
from hydro.hydro_reservoir import HydroReservoir
import constants


def test_zero_net_load():
    reservoir = HydroReservoir()
    net_load = np.zeros(shape=(constants.NB_HOURS, 1), dtype=np.float64)
    gain_function = HydroGainFunction(net_load, reservoir)
    for i in range(100):
        week = np.random.randint(0, constants.RESULTS_SIZE)
        control = np.random.uniform(0.1, 1)  # 1 : valeurs de max turb
        assert gain_function.get_gain(week, 0, 0) == pytest.approx(0.), "failed " + str(week) + " " + str(control)
        assert gain_function.get_gain(week, 0, control) < 0., "failed " + str(week) + " " + str(control)
        assert gain_function.get_gain(week, 0, -control) < 0., "failed " + str(week) + " " + str(control)


def test_random_net_load():
    capacity = 100
    w_max_turb = np.ones(shape=constants.RESULTS_SIZE, dtype=np.float64) * capacity/2
    w_max_pump = np.ones(shape=constants.RESULTS_SIZE, dtype=np.float64) * capacity/2
    h_max_turb = np.ones(shape=constants.NB_HOURS, dtype=np.float64) * capacity/2
    h_max_pump = np.ones(shape=constants.NB_HOURS, dtype=np.float64) * capacity/2
    reservoir = HydroReservoir(capacity=capacity,
                               weekly_max_pump=w_max_pump,
                               weekly_max_turb=w_max_turb,
                               hourly_max_pump=h_max_pump,
                               hourly_max_turb=h_max_turb,)
    net_load = (np.random.rand(constants.NB_HOURS, 1) - 0.5) * 20. / 168.
    gain_function = HydroGainFunction(net_load, reservoir)
    for week in range(constants.RESULTS_SIZE):
        weekly_net_load = net_load[week * 168:(week + 1) * 168, 0].sum()
        assert gain_function.get_gain(week, 0, weekly_net_load) > -1.5, (
                "failed " + str(gain_function.get_gain(week, 0, weekly_net_load))
        )
        # assert gain_function.get_gain(week, 0, control) < 0., "failed " + str(week) + " " + str(control)
        # assert gain_function.get_gain(week, 0, -control) < 0., "failed " + str(week) + " " + str(control)
