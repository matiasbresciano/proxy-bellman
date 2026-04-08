import pytest
import numpy as np

from hydro.proxy import HydroProxy
from hydro.reservoir import HydroReservoir
import constants


def test_zero_net_load():
    nb_sce = 1
    max_weekly_pump = np.ones(shape=constants.RESULTS_SIZE, dtype=np.float64)*10
    max_hourly_pump = np.ones(shape=constants.NB_HOURS, dtype=np.float64)*10 / (7 * 24)
    inflow = np.ones(shape=(constants.NB_HOURS, 1), dtype=np.float64) / 100
    reservoir = HydroReservoir(initial_level=50,
                               final_level=50,
                               hourly_max_pump=max_hourly_pump,
                               hourly_max_turb=max_hourly_pump,
                               weekly_max_pump=max_weekly_pump,
                               weekly_max_turb=max_weekly_pump,
                               hourly_inflow=inflow
                               )
    net_load = np.zeros(shape=(constants.NB_HOURS, nb_sce), dtype=np.float64)
    proxy = HydroProxy(net_load, [reservoir])
    ctrls = proxy.get_controls()
    print(ctrls)
