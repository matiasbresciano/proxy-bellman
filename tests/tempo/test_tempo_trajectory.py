import pytest
import numpy as np

from tempo.trajectory import TempoTrajectory
from tempo.bellman import TempoBellman
from tempo.cost_function import TempoCostFunction
from tempo.reservoir import TempoReservoir
import constants


def test_trajectories():
    nb_sce = 2
    scenarii = np.arange(nb_sce)
    mc_years = scenarii
    np.random.seed(0)
    residual_load = np.random.rand(constants.NB_DAYS + 1, nb_sce)*1000
    res = TempoReservoir(capacity=22)
    cost = TempoCostFunction(residual_load, res)
    bellman = TempoBellman(scenarii, cost, res)
    trajectory = TempoTrajectory(mc_years, res, cost, bellman)
    traj = trajectory.get_trajectories()
    for i in range(constants.RESULTS_SIZE):
        current_monday_idx = 7 * i + (7 - res.week_day_first_september) % 7
        if current_monday_idx < res.first_day:
            assert not np.any(traj[:, i] - 22)
        elif current_monday_idx > res.last_day:
            assert not np.any(traj[:, i])
        else:
            assert 0 <= traj[1, i] <= 22

