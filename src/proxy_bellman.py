import os
import time

import typer
from typing_extensions import Annotated

from tempo.proxy import TempoAntaresProxy
from hydro.proxy import HydroAntaresProxy



app = typer.Typer()


@app.command()
def tempo(
        dir_study: Annotated[str, typer.Argument(help="Antares study directory.")],
        areas: Annotated[list[str], typer.Argument(help="List of study areas (space-separated).")],
        mc_years: Annotated[int, typer.Option(help="Number of Monte-Carlo years to simulate.")] = 200,
        ts_selection: Annotated[str | None, typer.Option(help="List of TS to consider when calculating Bellman values, separated by coma, no space. Default is all TS.")] = None,
        dir_output: Annotated[str, typer.Option(help="Directory used for outputs.")] = ".",
        cvar: Annotated[float, typer.Option(help="CVaR parameter for trajectory generation.")] = 1.0,
        actions: Annotated[list[str], typer.Option(help="Actions to perform. Use --actions once for each action")] = ["None"]
) -> None:
    """
    Launch Tempo trajectories generation.
    Possible actions are: export_trajectories, export_daily_controls, export_calendar
    """
    if ts_selection:
        ts_selection = [int(a) for a in ts_selection.split(",")]

    for area in areas:
        print(f"Computing area {area}")
        proxy = TempoAntaresProxy(dir_study, area, mc_years, ts_selection, cvar)
        proxy.save_residual_loads()
        dir_output_area = os.path.join(dir_output, area)
        nb_sce = mc_years
        if ts_selection:
            nb_sce = len(ts_selection)
        for action in actions:
            match action:
                case "export_trajectories":
                    proxy.export_trajectories(dir_output_area)
                case "export_daily_controls":
                    proxy.export_controls(dir_output_area)
                case "export_calendar":
                    for s in range(nb_sce):
                        proxy.export_daily_controls(0, dir_output_area)
                case _:
                    print(f"Unknown action: {action}")


@app.command()
def hydro(
        dir_study: Annotated[str, typer.Argument(help="Antares study directory.")],
        areas: Annotated[list[str], typer.Argument(help="List of study areas (space-separated).")],
        mc_years: Annotated[int, typer.Option(help="Number of Monte-Carlo years to simulate.")] = 200,
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
    if ts_selection:
        ts_selection = [int(a) for a in ts_selection.split(",")]
    for area in areas:
        print(f"Computing area {area}")
        proxy = HydroAntaresProxy(dir_study, area, mc_years, ts_selection, nb_turb, alpha, penalty_factor)
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



if __name__ == '__main__':
    app()
