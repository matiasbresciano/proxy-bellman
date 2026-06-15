import pytest
import numpy as np

from tempo.bellman import TempoBellman
from tempo.cost_function import TempoCostFunction
from tempo.reservoir import TempoReservoir
import constants


def test_bellman_values():
    nb_sce = 2
    scenarii = np.arange(nb_sce)
    residual_load = np.random.rand(constants.NB_DAYS + 1, nb_sce)*1000
    res = TempoReservoir(capacity=22)
    cost = TempoCostFunction(residual_load, res)
    bellman = TempoBellman(scenarii, cost, res)
    b = bellman.get_bellman_values()
    for i in range(constants.RESULTS_SIZE):
        current_monday_idx = 7 * i + (7 - res.week_day_first_september) % 7
        if current_monday_idx < res.first_day or current_monday_idx > res.last_day:
            # on regarde les données de la semaine précédente à chaque fois parce que les valeurs de bellman
            # sont basées sur les couts de la semaine suivante
            assert not np.any(b[i-1]), "week : " + str(i)
        else:
            assert b[i-1, 1] > 0, "week : " + str(i)
