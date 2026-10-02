import os
from pathlib import Path
import datetime

import typer
from typing_extensions import Annotated

import numpy as np
import yaml

from tempo.proxy import TempoAntaresProxy
from hydro.proxy import HydroAntaresProxy

"""
Module for the command line.
"""


app = typer.Typer()


@app.command()
def tempo(
        dir_study: Annotated[str, typer.Argument(help="Antares study directory.")],
        areas: Annotated[list[str], typer.Argument(help="List of study areas (space-separated).")],
        mc_years: Annotated[str, typer.Option(help="Number of Monte-Carlo years to simulate.")] = "200",
        ts_selection: Annotated[str | None, typer.Option(help="List of TS to consider when calculating Bellman values, separated by coma, no space. Default is all TS.")] = None,
        dir_output: Annotated[str, typer.Option(help="Directory used for outputs.")] = ".",
        actions: Annotated[list[str], typer.Option(help="Actions to perform. Use --actions once for each action from [export_trajectories, export_controls, ...]")] = ["None"],
        host: Annotated[str, typer.Option(help="Antares web host.")] = "",
        token: Annotated[str, typer.Option(help="Antares web token.")] = "",
        study_id: Annotated[str, typer.Option(help="Antares web study id.")] = ""
) -> None:
    """
    Launch Tempo trajectories generation.
    Possible actions are: export_trajectories, export_daily_controls, export_calendar
    """
    ts_selection_list = parse_years(ts_selection)
    mc_years_list = parse_years(mc_years)

    assert mc_years_list is not None

    for area in areas:
        print(f"Computing area {area}")
        proxy = TempoAntaresProxy(dir_study, area, mc_years_list, ts_selection_list, host, token, study_id)
        dir_output_area = os.path.join(dir_output + datetime.datetime.now().strftime("_%Y-%m-%d-%H-%M-%S")
, area)
        print(f"Results for this area are exported in {dir_output_area}.")
        for action in actions:
            match action:
                case "export_trajectories":
                    proxy.export_trajectories(dir_output_area)
                case "export_controls":
                    proxy.export_controls(dir_output_area)
                case "export_calendar":
                    for s in mc_years_list:
                        proxy.export_daily_controls(0, dir_output_area)
                case _:
                    print(f"Unknown action: {action}")


@app.command()
def hydro(
        dir_study: Annotated[str, typer.Argument(help="Antares study directory.")],
        areas: Annotated[list[str], typer.Argument(help="List of study areas (space-separated).")],
        mc_years: Annotated[str, typer.Option(help="Number of Monte-Carlo years to simulate.")] = "200",
        ts_selection: Annotated[str | None, typer.Option(help="List of TS to consider when calculating Bellman values, separated by coma, no space. Default is all TS.")] = None,
        dir_output: Annotated[str, typer.Option(help="Directory used for outputs.")] = "./results",
        nb_turb: Annotated[int, typer.Option(help="Number of values on which to compute the cost function.")] = 25,
        alpha: Annotated[int, typer.Option(help="parameter for the computation of the costs value and the turbine vs pumping ratio")] = 2,
        penalty_factor: Annotated[float, typer.Option(help="factor to modulate how important it is to respect guidelines")] = 1,
        actions: Annotated[list[str], typer.Option(help="Actions to perform. Use --actions once for each action")] = ["None"],
        tmp_dir: Annotated[str, typer.Option(help="directory for putting back up files when modifying the study, or when to fetch them for rolling back")] = "./",
        host: Annotated[str, typer.Option(help="Antares web host.")] = "",
        token: Annotated[str, typer.Option(help="Antares web token.")] = "",
        study_id: Annotated[str, typer.Option(help="Antares web study id.")] = ""
) -> None:
    """
    Launch the generation of storage trajectories for one or multiple areas.
    Possible actions are: export_trajectories, export_controls, modify_antares_data, undo_modifications
    """
    ts_selection_list = parse_years(ts_selection)
    mc_years_list = parse_years(mc_years)

    assert mc_years_list is not None

    for area in areas:
        print(f"Computing area {area}")
        proxy = HydroAntaresProxy(dir_study, area, mc_years_list, ts_selection_list, nb_turb, alpha, penalty_factor, tmp_dir, host, token, study_id)
        dir_output_area = os.path.join(dir_output + datetime.datetime.now().strftime("_%Y-%m-%d-%H-%M-%S"), area)
        print(f"Results for this area are exported in {dir_output_area}.")
        for action in actions:
            match action:
                case "export_controls":
                    proxy.export_controls(dir_output_area)
                case "export_trajectories":
                    proxy.export_trajectories(dir_output_area)
                case "modify_antares_data":
                    proxy.apply_to_study()
                case "undo_modifications":
                    proxy.undo_study()
                case _:
                    print(f"Unknown action: {action}")

@app.command()
def yaml_settings(settings_path: Annotated[str, typer.Argument(help="Yaml settings path.")]) -> None:
    with Path(settings_path).open() as file:
        settings_yaml = yaml.safe_load(file)
        if "hydro" in settings_yaml:
            hydro_settings = settings_yaml["hydro"]

            dir_study = ""
            host = ""
            study_id = ""
            if "study" in hydro_settings:
                dir_study = hydro_settings["study"]
            else:
                host = hydro_settings["host"]
                token = ""
                if "token" in hydro_settings:
                    token = hydro_settings["token"]
                study_id = hydro_settings["study_id"]
            areas = hydro_settings["areas"]
            output_dir = "./results"
            if "output_dir" in hydro_settings:
                output_dir = hydro_settings["output_dir"]
            tmp_dir = "./"
            if "tmp_dir" in hydro_settings:
                tmp_dir = hydro_settings["tmp_dir"]
            mc_years = "200"
            if "mc_years" in hydro_settings:
                mc_years = hydro_settings["mc_years"]
            ts_selection = None
            if "ts_selection" in hydro_settings:
                ts_selection = hydro_settings["ts_selection"]
            nb_turb = 25
            if "nb_turb" in hydro_settings:
                nb_turb = hydro_settings["nb_turb"]
            alpha = 2
            if "alpha" in hydro_settings:
                alpha = hydro_settings["alpha"]
            penalty_factor = 1
            if "penalty_factor" in hydro_settings:
                penalty_factor = hydro_settings["penalty_factor"]
            actions = ["None"]
            if "actions" in hydro_settings:
                actions = hydro_settings["actions"]
            hydro(dir_study, areas, mc_years, ts_selection, output_dir, nb_turb, alpha, penalty_factor, actions, tmp_dir, host, token, study_id)

        elif "tempo" in settings_yaml:
            tempo_settings = settings_yaml["tempo"]
            dir_study = ""
            host = ""
            study_id = ""
            if "study" in tempo_settings:
                dir_study = tempo_settings["study"]
            else:
                host = tempo_settings["host"]
                token = ""
                if "token" in tempo_settings:
                    token = tempo_settings["token"]
                study_id = tempo_settings["study_id"]
            areas = tempo_settings["areas"]
            output_dir = "./results"
            if "output_dir" in tempo_settings:
                output_dir = tempo_settings["output_dir"]
            mc_years = "200"
            if "mc_years" in tempo_settings:
                mc_years = tempo_settings["mc_years"]
            ts_selection = None
            if "ts_selection" in tempo_settings:
                ts_selection = tempo_settings["ts_selection"]
            actions = ["None"]
            if "actions" in tempo_settings:
                actions = tempo_settings["actions"]
            tempo(dir_study, areas, mc_years, ts_selection, output_dir, actions, host, token, study_id)


def parse_years(years: str | None) -> np.ndarray | None:
    if not years:
        res = None
    elif years.find(":") != -1:
        borns = [int(a) for a in years.split(":")]
        assert len(
            borns) == 2, f"In range mode must comport exactly 2 values. {len(borns)} were provided."
        assert borns[0] < borns[
            1], f"In range mode, first value of must be strictly inferior to second value."
        res = np.arange(borns[0], borns[1])
    elif years.find(",") != -1:
        res = np.asarray([int(a) for a in years.split(",")])
    else:
        res = np.arange(int(years))
    return res


if __name__ == '__main__':
    app()
