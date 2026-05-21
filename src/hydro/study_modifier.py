import os
import shutil
import numpy as np
from pathlib import Path
import pandas as pd
from configparser import ConfigParser
import antares.craft as ac
from antares.craft.model.st_storage import STStorage

from hydro.reservoir import HydroReservoir
from hydro.trajectory import HydroTrajectory
import constants


class StudyModifier:
    def __init__(self, nb_sce: int, res: HydroReservoir, trajectories: HydroTrajectory,
                 study_path: str, name_area: str):
        """
        Initialize the class with BellmanValuesProxy, optimal trajectories, and the target area.
        """
        self.reservoir = res
        self.trajectories = trajectories
        self.nb_scenarios = nb_sce
        self.study_path = study_path
        self.study = ac.read_study_local(Path(study_path))
        self.area_name = name_area
        self.area = self.study.get_areas()[self.area_name]
        self.storage = None

    def apply_all(self) -> None:
        """
        Execute all the steps to modify the Antares study.
        """
        self.overwrite_pmax()
        self.create_st_cluster()
        self.create_pmax_file()
        self.modify_scenario_builder()
        self.create_inflows_sts()
        self.adjust_to_spillage_constraint()

    def undo_all(self) -> None:
        """
        Perform the full restoration of the Antares study for the original area.
        """
        self.restore_pmax()
        self.remove_st_cluster_section()
        self.clean_scenariobuilder()
        self.restore_miscgen_load_and_solar()

    def overwrite_pmax(self) -> None:
        """
        Replace the pmax file (maxpower_{area}.txt) with a file where all values are zero,
        backing up the original file first.
        """
        max_power = self.area.hydro.get_maxpower()

        pmax_path = os.path.join(self.study_path, "input", "hydro", "common", "capacity"
                                 , f"maxpower_{self.area_name}.txt")
        pmax_backup_path = pmax_path.replace(".txt", "_old.txt")

        if os.path.exists(pmax_path):
            shutil.copy(pmax_path, pmax_backup_path)

        max_power[0] = 0
        max_power[2] = 0

        self.area.hydro.set_maxpower(max_power)

    def create_st_cluster(self) -> None:
        """
        Append a section to the list.ini file defining an ST storage cluster,
        including its capacities and efficiencies.
        """
        prop = ac.STStorageProperties(
            group="PSP_open",
            reservoir_capacity=self.reservoir.capacity,
            initial_level=0.5,
            injection_nominal_capacity=np.max(self.reservoir.hourly_max_pump),
            withdrawal_nominal_capacity=np.max(self.reservoir.hourly_max_turb),
            efficiency=self.reservoir.pump_efficiency,
            efficiency_withdrawal=self.reservoir.turb_efficiency,
            initial_level_optim=False,
            enabled=True
        )
        self.storage = self.area.create_st_storage(f"""lt_stock_proxy_{self.area_name}""", prop)

    def create_pmax_file(self) -> None:
        """
        Generate PMAX-injection.txt and PMAX-withdrawal.txt files for the area,
        based on maximum hourly pumping and turbine capacities,
        concatenated with 24 additional values.
        """
        assert isinstance(self.storage, STStorage)
        # pmax_injection_hourly = self.storage.get_pmax_injection()
        # pmax_withdrawal_hourly = self.storage.get_pmax_withdrawal()

        pmax_injection_hourly = self.reservoir.hourly_max_pump
        max_injection = np.max(pmax_injection_hourly[0])
        if max_injection != 0:
            pmax_injection_hourly = pmax_injection_hourly / max_injection

        pmax_injection_hourly = np.concatenate((pmax_injection_hourly, pmax_injection_hourly[-24:]))


        pmax_withdrawal_hourly = self.reservoir.hourly_max_turb
        max_withdrawal = np.max(pmax_withdrawal_hourly[0])
        if max_withdrawal != 0:
            pmax_withdrawal_hourly = pmax_withdrawal_hourly / max_withdrawal

        pmax_withdrawal_hourly = np.concatenate((pmax_withdrawal_hourly, pmax_withdrawal_hourly[-24:]))

        self.storage.set_pmax_withdrawal(pd.DataFrame(pmax_withdrawal_hourly))
        self.storage.set_pmax_injection(pd.DataFrame(pmax_injection_hourly))

    def modify_scenario_builder(self) -> None:
        """
        Create a text file in user/tmp/scenariobuilder_lines listing
        the lines needed to assign ST clusters and to MC scenarios.
        """
        config = ConfigParser(strict=False)
        config.read(os.path.join(self.study_path, "settings", "generaldata.ini"))
        nbyears = int(config["general"]["nbyears"])

        lines = []
        for mc in range(nbyears):
            trajectory = (mc % self.nb_scenarios) + 1
            lines.append(f"sts,{self.area_name},{mc},lt_stock_proxy_{self.area_name}={trajectory}")
            lines.append(f"s,{self.area_name},{mc},lt_stock_proxy_{self.area_name}={trajectory}")

        sb_dir = os.path.join(self.study_path, "user", "tmp", "scenariobuilder_lines")
        os.makedirs(sb_dir, exist_ok=True)
        with open(os.path.join(sb_dir, f"{self.area_name}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

    def adjust_inflow_pmax_withdrawal_constraint(self, balance: np.ndarray, week: int) -> np.ndarray:
        """
        Adjust the hourly balance at the end of the week to not exceed
        the maximum weekly turbine capacity (avoid numerical rounding errors leading to infeasibilities).
        """
        delta = np.sum(balance) - np.sum(self.reservoir.weekly_max_turb[week] * self.reservoir.turb_efficiency)
        if delta > 0:
            balance[-1] -= np.ceil(delta / 1e-6) * 1e-6
        return balance

    def adjust_inflows_pmax_injection_constraint(self, balance: np.ndarray, week: int) -> np.ndarray:
        """
        Adjust the hourly balance at the end of the week to not exceed
        the maximum weekly pumping capacity (avoid numerical rounding errors leading to infeasibilities).
        """
        delta = np.sum(balance) + np.sum(self.reservoir.weekly_max_pump[week] * self.reservoir.pump_efficiency)
        if delta < 0:
            balance[-1] -= np.floor(delta / 1e-6) * 1e-6
        return balance

    def create_inflows_sts(self) -> None:
        """
        Generate inflows.txt for the ST proxy by calculating the adjusted hourly balance
        according to constraints for each scenario and week.
        """
        assert isinstance(self.storage, STStorage)
        inflow = self.storage.get_storage_inflows()
        balance = np.zeros((constants.NB_HOURS, self.nb_scenarios))
        for s in range(self.nb_scenarios):
            for w in range(constants.RESULTS_SIZE):
                hour_start = w * 168
                if w == 0:
                    hlevel_start = self.reservoir.initial_level
                else:
                    hlevel_start = self.trajectories.get_trajectories()[s, w - 1]
                hlevel_end = self.trajectories.get_trajectories()[s, w]

                # This computes optimal control over the week :
                # balance[hour_start, s] + balance[hour_start + 167, s] = hlevel_start - hlevel_end
                balance[hour_start, s] = hlevel_start - self.reservoir.capacity / 2
                balance[hour_start + 167, s] = self.reservoir.capacity / 2 - hlevel_end

                # Add hourly inflow to amount to be balanced
                hourly_inflow = self.reservoir.hourly_inflow[hour_start:hour_start + 168, s]
                balance[hour_start:hour_start + 168, s] += hourly_inflow

                # Adjust inflows to respect st storage constraints (overflow and negative
                # stock leading to unfeasibilites)
                balance[hour_start:hour_start + 168, s] -= self.trajectories.inflow_adjust_overflow[w, s, :]

                # Adjust inflows to respect pmax constraints (avoid numerical rounding errors
                # leading to infeasibilities)
                balance[hour_start:hour_start + 168, s] = self.adjust_inflow_pmax_withdrawal_constraint(
                    balance[hour_start:hour_start + 168, s], w
                )
                balance[hour_start:hour_start + 168, s] = self.adjust_inflows_pmax_injection_constraint(
                    balance[hour_start:hour_start + 168, s], w
                )

                # Final check
                if (
                        np.sum(balance[hour_start:hour_start + 168, s])
                        > self.reservoir.weekly_max_turb[w] * self.reservoir.turb_efficiency
                        or np.sum(balance[hour_start:hour_start + 168, s])
                        < -self.reservoir.weekly_max_pump[w] * self.reservoir.pump_efficiency
                ):
                    raise ValueError(
                        f"Error for area {self.area_name} in week {w} scenario {s}: balance: {np.sum(balance[hour_start:hour_start + 168, s])}, "
                        f"max turbine: {self.reservoir.weekly_max_turb[w] * self.reservoir.turb_efficiency}, "
                        f"max pump: {-self.reservoir.weekly_max_pump[w] * self.reservoir.pump_efficiency}"
                    )

        self.storage.set_storage_inflows(pd.DataFrame(balance))

    def adjust_to_spillage_constraint(self) -> None:
        """
        Complies with spillage constraint. Negative net load is transfered to solar producion and
        max(turbining capacity,pumping capacity) is added to net load and misc-gen (fatal production).
        """
        miscgen_path = os.path.join(self.study_path, "input", "misc-gen", f"miscgen-{self.area_name}.txt")
        load_path = os.path.join(self.study_path, "input", "load", "series", f"load_{self.area_name}.txt")
        solar_path = os.path.join(self.study_path, "input", "solar", "series", f"solar_{self.area_name}.txt")

        miscgen_backup_path = miscgen_path.replace(".txt", "_old.txt")
        load_backup_path = load_path.replace(".txt", "_old.txt")
        solar_backup_path = solar_path.replace(".txt", "_old.txt")

        s = self.nb_scenarios

        miscgen_data = self.area.get_misc_gen_matrix().values
        miscgen_data = np.asarray(miscgen_data, dtype=np.float64)
        load_data = self.area.get_load_matrix().values
        solar_data = self.area.get_solar_matrix().values

        # On fait les back-up
        if os.path.exists(miscgen_path):
            if not os.path.exists(miscgen_backup_path):
                shutil.copy(miscgen_path, miscgen_backup_path)

        if os.path.exists(load_path):
            if not os.path.exists(load_backup_path):
                shutil.copy(load_path, load_backup_path)

        if os.path.exists(solar_path):
            if not os.path.exists(solar_backup_path):
                shutil.copy(solar_path, solar_backup_path)

        negatives = np.minimum(load_data, 0.0)
        transfer = -negatives
        load_data = load_data - negatives
        solar_data = solar_data + transfer

        hourly_turb = self.reservoir.hourly_max_turb
        hourly_pump = self.reservoir.hourly_max_pump
        spill_constraint = np.maximum(hourly_turb, hourly_pump)
        spill_constraint = np.concatenate([spill_constraint, spill_constraint[-24:]])

        miscgen_data[:, 5] += spill_constraint[:8760]
        load_data = load_data + spill_constraint[:8760, np.newaxis]

        self.area.set_misc_gen(pd.DataFrame(miscgen_data))
        self.area.set_load(pd.DataFrame(load_data))
        self.area.set_solar(pd.DataFrame(solar_data))

    def restore_pmax(self) -> None:
        """
        Restore the pmax file (maxpower_area.txt) by replacing the current version
        with the backup (_old.txt) if it exists.
        """
        pmax_path = os.path.join(
            self.study_path, "input", "hydro", "common", "capacity", f"maxpower_{self.area_name}.txt"
        )
        pmax_backup_path = pmax_path.replace(".txt", "_old.txt")
        if os.path.exists(pmax_backup_path):
            if os.path.exists(pmax_path):
                os.remove(pmax_path)
            shutil.copy(pmax_backup_path, pmax_path)
            os.remove(pmax_backup_path)

    def remove_st_cluster_section(self) -> None:
        """
        Remove the ST proxy section from the storage cluster list.ini file
        for the area.
        """
        try:
            storage = self.area.get_st_storages()[f"""lt_stock_proxy_{self.area_name}"""]
            self.area.delete_st_storage(storage)
        except KeyError:
            pass

    def clean_scenariobuilder(self) -> None:
        """
        Clean the scenariobuilder.dat file by removing lines associated with
        the ST proxy for the area (both 'sts' and 's' entries).
        """
        pass
        # TODO à refaire si l'étude n'est pas la même après l'undo

    def restore_miscgen_load_and_solar(self) -> None:
        """
        Restore study inputs to their pre-modification state:
        - Restore miscgen-{area}.txt from miscgen-{area}_old.txt if it exists.
        - Restore load_{area}.txt    from load_{area}_old.txt    if it exists.
        - Restore solar_{area}.txt   from solar_{area}_old.txt   if it exists; otherwise remove solar_{area}.txt
        (this file may have been created when negative load was transferred to solar per scenario).
        Missing backups are ignored; existing current files are overwritten or removed as needed.
        """
        miscgen_path = os.path.join(self.study_path, "input", "misc-gen", f"miscgen-{self.area_name}.txt")
        miscgen_backup_path = miscgen_path.replace(".txt", "_old.txt")
        if os.path.exists(miscgen_backup_path):
            if os.path.exists(miscgen_path):
                os.remove(miscgen_path)
            shutil.copy(miscgen_backup_path, miscgen_path)
            os.remove(miscgen_backup_path)

        load_path = os.path.join(self.study_path, "input", "load", "series", f"load_{self.area_name}.txt")
        load_backup_path = load_path.replace(".txt", "_old.txt")
        if os.path.exists(load_backup_path):
            if os.path.exists(load_path):
                os.remove(load_path)
            shutil.copy(load_backup_path, load_path)
            os.remove(load_backup_path)

        solar_path = os.path.join(self.study_path, "input", "solar", "series", f"solar_{self.area_name}.txt")
        solar_backup_path = solar_path.replace(".txt", "_old.txt")
        if os.path.exists(solar_backup_path):
            if os.path.exists(solar_path):
                os.remove(solar_path)
            shutil.copy(solar_backup_path, solar_path)
            os.remove(solar_backup_path)
