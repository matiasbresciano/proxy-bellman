import pytest
import numpy as np
import math

from hydro.hydro_cost_function import HydroCostFunction
from hydro.hydro_reservoir import HydroReservoir
import constants


def test_zero_net_load():
    reservoir = HydroReservoir()
    net_load = np.zeros(shape=(constants.NB_HOURS, 1), dtype=np.float64)
    cost_function = HydroCostFunction(net_load, reservoir)
    for i in range(100):
        week = np.random.randint(0, constants.RESULTS_SIZE)
        control = np.random.uniform(0.1, 1)  # 1 : valeurs de max turb
        assert cost_function.get_cost(week, 0, 0) == pytest.approx(0.), "failed " + str(week) + " " + str(control)
        assert cost_function.get_cost(week, 0, control) < 0., "failed " + str(week) + " " + str(control)
        assert cost_function.get_cost(week, 0, -control) < 0., "failed " + str(week) + " " + str(control)


def test_random_net_load():
    capacity = 100
    w_max_turb = np.ones(shape=constants.RESULTS_SIZE, dtype=np.float64) * capacity
    w_max_pump = np.ones(shape=constants.RESULTS_SIZE, dtype=np.float64) * capacity
    h_max_turb = np.ones(shape=constants.NB_HOURS, dtype=np.float64) * capacity / (7*24)
    h_max_pump = np.ones(shape=constants.NB_HOURS, dtype=np.float64) * capacity / (7*24)
    reservoir = HydroReservoir(capacity=capacity,
                               weekly_max_pump=w_max_pump,
                               weekly_max_turb=w_max_turb,
                               hourly_max_pump=h_max_pump,
                               hourly_max_turb=h_max_turb)
    net_load = (np.random.rand(constants.NB_HOURS, 1) - 0.5) * 20. / 168.
    cost_function = HydroCostFunction(net_load, reservoir, turb_threshold=1000)
    for week in range(constants.RESULTS_SIZE):
        weekly_net_load = net_load[week * 168:(week + 1) * 168, 0].sum()
        assert math.floor(cost_function.get_cost(week, 0, weekly_net_load)) < 0.001, (
                "failed " + str(cost_function.get_cost(week, 0, weekly_net_load))
        )
