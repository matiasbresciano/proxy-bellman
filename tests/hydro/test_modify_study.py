import os
import pytest

from hydro.proxy import HydroAntaresProxy


dir_study = "test_data/two_nodes"

back_up_hydro_filepath = os.path.join(dir_study, "input", "hydro", "common", "capacity", "maxpower_area1_old.txt")
miscgen_backup_path = os.path.join(dir_study, "input", "misc-gen", "miscgen-area1_old.txt")
load_backup_path = os.path.join(dir_study, "input", "load", "series", "load_area1_old.txt")
solar_path = os.path.join(dir_study, "input", "solar", "series", "solar_area1_old.txt")

loads_filepath = os.path.join(dir_study, "user", "residual_loads.txt")

def test_modify():
    assert not os.path.exists(back_up_hydro_filepath)
    assert not os.path.exists(miscgen_backup_path)
    assert not os.path.exists(load_backup_path)
    assert not os.path.exists(solar_path)
    area1 = "area1"
    proxy1 = HydroAntaresProxy(dir_study, area1, 10, turb_threshold=10, alpha=2)
    proxy1.apply_to_study()
    assert os.path.exists(back_up_hydro_filepath)
    assert os.path.exists(miscgen_backup_path)
    assert os.path.exists(load_backup_path)
    assert os.path.exists(solar_path)

    os.remove(loads_filepath)


def test_undo():
    assert os.path.exists(back_up_hydro_filepath)
    assert os.path.exists(miscgen_backup_path)
    assert os.path.exists(load_backup_path)
    assert os.path.exists(solar_path)
    area1 = "area1"
    proxy1 = HydroAntaresProxy(dir_study, area1, 10, turb_threshold=10, alpha=2)
    proxy1.undo_study()
    assert not os.path.exists(back_up_hydro_filepath)
    assert not os.path.exists(miscgen_backup_path)
    assert not os.path.exists(load_backup_path)
    assert not os.path.exists(solar_path)
    os.remove(loads_filepath)


def test_write_loads():
    assert not os.path.exists(loads_filepath)
    area1 = "area1"
    proxy1 = HydroAntaresProxy(dir_study, area1, 10, turb_threshold=10, alpha=2)
    proxy1.save_residual_loads()
    assert os.path.exists(loads_filepath)
    os.remove(loads_filepath)

