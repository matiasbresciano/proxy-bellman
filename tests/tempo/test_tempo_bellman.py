import pytest
import numpy as np

from tempo.bellman import TempoBellman
from tempo.cost_function import TempoCostFunction
from tempo.reservoir import TempoReservoir
import constants


def test_bellman_values():
    nb_sce = 2
    residual_load = np.random.rand(constants.NB_DAYS + 1, nb_sce)*1000
    res = TempoReservoir(capacity=22)
    cost = TempoCostFunction(residual_load, res)
    bellman = TempoBellman(nb_sce, cost, res)
    b = bellman.get_bellman_values()
    for i in range(constants.RESULTS_SIZE):
        if res.get_previous_monday(7*i) + 6 < res.first_day or res.get_previous_monday(7*i) > res.last_day:
            assert not np.any(b[i])
        else:
            assert b[i, 1] > 0
