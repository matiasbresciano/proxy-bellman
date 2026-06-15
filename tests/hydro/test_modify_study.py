import os
import pytest
import numpy as np

from hydro.proxy import HydroAntaresProxy


dir_study = "test_data/two_nodes"
area1 = "area1"

back_up_hydro_filepath = os.path.join(dir_study, "input", "hydro", "common", "capacity", "maxpower_area1_old.txt")
miscgen_backup_path = os.path.join(dir_study, "input", "misc-gen", "miscgen-area1_old.txt")
load_backup_path = os.path.join(dir_study, "input", "load", "series", "load_area1_old.txt")
solar_path = os.path.join(dir_study, "input", "solar", "series", "solar_area1_old.txt")

loads_filepath = os.path.join(dir_study, "user", "residual_loads.txt")


def test_modify_and_undo():
    if os.path.exists(back_up_hydro_filepath):
        os.remove(back_up_hydro_filepath)
    if os.path.exists(miscgen_backup_path):
        os.remove(miscgen_backup_path)
    if os.path.exists(load_backup_path):
        os.remove(load_backup_path)
    if os.path.exists(solar_path):
        os.remove(solar_path)
    proxy1 = HydroAntaresProxy(dir_study, area1, np.arange(10), turb_threshold=10, alpha=2)
    proxy1.apply_to_study()
    assert os.path.exists(back_up_hydro_filepath)
    assert os.path.exists(miscgen_backup_path)
    assert os.path.exists(load_backup_path)
    assert os.path.exists(solar_path)
    proxy1.undo_study()
    assert not os.path.exists(back_up_hydro_filepath)
    assert not os.path.exists(miscgen_backup_path)
    assert not os.path.exists(load_backup_path)
    assert not os.path.exists(solar_path)


def test_write_loads():
    if os.path.exists(loads_filepath):
        os.remove(loads_filepath)
    proxy1 = HydroAntaresProxy(dir_study, area1, np.arange(10), turb_threshold=10, alpha=2)
    proxy1.save_residual_loads()
    assert os.path.exists(loads_filepath)
    os.remove(loads_filepath)


export_dir = os.path.join(dir_study, "user")


def test_export_trajectories():
    proxy1 = HydroAntaresProxy(dir_study, area1, np.arange(10), turb_threshold=10, alpha=2)
    traj_path = os.path.join(export_dir, "trajectories.csv")
    if os.path.exists(traj_path):
        os.remove(traj_path)
    proxy1.export_trajectories(export_dir)
    assert os.path.exists(traj_path)


def test_export_controls():
    proxy1 = HydroAntaresProxy(dir_study, area1, np.arange(10), turb_threshold=10, alpha=2)
    controls_path = os.path.join(export_dir, "controls.csv")
    if os.path.exists(controls_path):
        os.remove(controls_path)
    proxy1.export_controls(export_dir)
    assert os.path.exists(controls_path)

