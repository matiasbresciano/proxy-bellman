import os
import time

import typer
from typing_extensions import Annotated

import numpy as np

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
        cvar: Annotated[float, typer.Option(help="CVaR parameter for trajectory generation.")] = 1.0,
        actions: Annotated[list[str], typer.Option(help="Actions to perform. Use --actions once for each action")] = ["None"]
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
        proxy = TempoAntaresProxy(dir_study, area, mc_years_list, ts_selection_list, cvar)
        proxy.save_residual_loads()
        dir_output_area = os.path.join(dir_output, area)
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
        dir_output: Annotated[str, typer.Option(help="Directory used for outputs.")] = ".",
        nb_turb: Annotated[int, typer.Option(help="Number of values on which to compute the cost function.")] = 25,
        alpha: Annotated[int, typer.Option(help="parameter for the computation of the costs value and the turbine vs pumping ratio")] = 2,
        penalty_factor: Annotated[float, typer.Option(help="factor to modulate how important it is to respect guidelines")] = 1,
        actions: Annotated[list[str], typer.Option(help="Actions to perform. Use --actions once for each action")] = ["None"]
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
        proxy = HydroAntaresProxy(dir_study, area, mc_years_list, ts_selection_list, nb_turb, alpha, penalty_factor)
        proxy.save_residual_loads()
        dir_output_area = os.path.join(dir_output, "area")
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
