import pytest
import numpy as np

from hydro.bellman import HydroBellman
from hydro.hydro_cost_function import HydroCostFunction
from hydro.hydro_reservoir import HydroReservoir
import constants


def test_zero_net_load():
    nb_sce = 1
    reservoir = HydroReservoir()
    net_load = np.zeros(shape=(constants.NB_HOURS, nb_sce), dtype=np.float64)
    cost_function = HydroCostFunction(net_load, reservoir)
    bellman = HydroBellman(nb_sce, 1., cost_function, reservoir)
    values = bellman.get_bellman_values()
    assert isinstance(values, np.ndarray)
    print(bellman.get_usage_values())
