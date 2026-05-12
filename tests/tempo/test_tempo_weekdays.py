import numpy as np

from tempo.proxy import TempoProxy
from tempo.reservoir import TempoReservoir


def test_january_first_month():
    net_load_sce = np.arange(365, dtype=np.float64)
    net_load = net_load_sce[:, np.newaxis]
    res_red = TempoReservoir()
    res_white_and_red = TempoReservoir(first_day=0, last_day=364)
    proxy = TempoProxy(net_load, [res_red, res_white_and_red], 0, 3)
    assert res_red.week_day_first_september == 0
    assert proxy._residual_load[0, 0] == 242  # 1er september a les données du 31 août
    first_marsh = res_red.day_of_year_from_september(0, 2)[0]
    assert proxy._residual_load.shape == (364, 1)
    assert proxy._residual_load[first_marsh, 0] == 59  # 1st Marsh must stay 1st Marsh
    first_january = res_red.day_of_year_from_september(0, 0)[0]
    assert proxy._residual_load[first_january, 0] == 0  # 1st january must stay 1st january
    assert proxy._residual_load[first_january - 1, 0] == 363  # 31st december has data from 30th december


def test_july_first_month():
    net_load_sce = np.arange(365, dtype=np.float64)
    first_july_from_january = 31+28+31+30+31+30
    net_load_sce = np.roll(net_load_sce, -first_july_from_january)  # first day is first july
    net_load = net_load_sce[:, np.newaxis]
    res_red = TempoReservoir()
    res_white_and_red = TempoReservoir(first_day=0, last_day=364)
    proxy = TempoProxy(net_load, [res_red, res_white_and_red], 6, 4)
    assert res_red.week_day_first_september == 1
    assert proxy._residual_load[0, 0] == 243  # 1er september a les données du 1er septembre
    first_marsh = res_red.day_of_year_from_september(0, 2)[0]
    assert proxy._residual_load.shape == (364, 1)
    assert proxy._residual_load[first_marsh, 0] == 59  # 1st Marsh must stay 1st Marsh
    first_january = res_red.day_of_year_from_september(0, 0)[0]
    assert proxy._residual_load[first_january, 0] == 0  # 1st january must stay 1st january
    assert proxy._residual_load[- 1, 0] == 242  # 30th August has data from 31st August
    first_july = res_red.day_of_year_from_september(0, 6)[0]
    assert proxy._residual_load[first_july, 0] == first_july_from_january + 1  # 1st july has data from 2nd July
    assert proxy._residual_load[first_july - 1, 0] == first_july_from_january   # 30th june has data from 1st July
    assert proxy._residual_load[first_july - 2, 0] == first_july_from_january - 2  # 29th june must stay 29th june


def test_marsh_first_month():
    net_load_sce = np.arange(365, dtype=np.float64)
    first_marsh_from_january = 31+28
    net_load_sce = np.roll(net_load_sce, -first_marsh_from_january)  # first day is first july
    net_load = net_load_sce[:, np.newaxis]
    res_red = TempoReservoir()
    res_white_and_red = TempoReservoir(first_day=0, last_day=364)
    proxy = TempoProxy(net_load, [res_red, res_white_and_red], 2, 3)
    assert res_red.week_day_first_september == 6
    assert proxy._residual_load[0, 0] == 242  # 1er september a les données du 31 août
    first_marsh = res_red.day_of_year_from_september(0, 2)[0]
    assert proxy._residual_load.shape == (364, 1)
    assert proxy._residual_load[first_marsh, 0] == 59  # 1st Marsh must stay 1st Marsh
    assert proxy._residual_load[first_marsh - 1, 0] == 57  # 28th february has data from 27th february
    first_january = res_red.day_of_year_from_september(0, 0)[0]
    assert proxy._residual_load[first_january, 0] == 364  # 1st january has data from 31st December
    assert proxy._residual_load[- 1, 0] == 241  # 30th August has data from 30th August


def test_april_first_month():
    net_load_sce = np.arange(365, dtype=np.float64)
    first_april_from_january = 31+28+31
    net_load_sce = np.roll(net_load_sce, -first_april_from_january)  # first day is first july
    net_load = net_load_sce[:, np.newaxis]
    res_red = TempoReservoir()
    res_white_and_red = TempoReservoir(first_day=0, last_day=364)
    proxy = TempoProxy(net_load, [res_red, res_white_and_red], 3, 3)
    assert res_red.week_day_first_september == 0
    assert proxy._residual_load[0, 0] == 243  # 1er septembre a les données du 1er septembre
    first_marsh = res_red.day_of_year_from_september(0, 2)[0]
    assert proxy._residual_load.shape == (364, 1)
    assert proxy._residual_load[first_marsh, 0] == 59  # 1st Marsh must stay 1st Marsh
    assert proxy._residual_load[first_marsh+30, 0] == 59+30  # 31st Marsh must stay 31st Marsh
    assert proxy._residual_load[first_marsh+31, 0] == 59+32  # 1st April has data from 2nd April
    first_january = res_red.day_of_year_from_september(0, 0)[0]
    assert proxy._residual_load[first_january, 0] == 0  # 1st january has data from 1st january
    assert proxy._residual_load[- 1, 0] == 242  # 30th August has data from 31st August
