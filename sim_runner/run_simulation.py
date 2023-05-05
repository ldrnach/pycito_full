import argparse
import json
from pathlib import Path

import drake_simulation.a1_simulator as a1_simulator
import drake_simulation.analysistools as analysistools
from configuration.build_from_config import build_from_config
from configuration.simulator import DrakeSimulatorConfig
from drake_simulation.exceptions import SimulationFailedException


def run_simulation(config: DrakeSimulatorConfig, savedir: Path) -> None:
    simbuilder: a1_simulator.A1DrakeSimulationBuilder = build_from_config(
        module=a1_simulator, config=config
    )

    simbuilder.set_initial_state(simbuilder.get_a1_standing_state())

    simbuilder.controller.enable_logging()

    try:
        simbuilder.initialize_sim()
        simbuilder.run_simulation(end_time=config.end_time)
    except SimulationFailedException:
        savedir = insert_into_path(savedir, "fail")
    else:
        savedir = insert_into_path(savedir, "success")

    simplotter = a1_simulator.A1SimulationPlotter()
    simplotter.plot(
        simbuilder.get_logs(), show=False, savename=str(savedir + "sim.png")
    )
    simplotter.save_data(simbuilder.get_logs(), savename=str(savedir + "simdata.pkl"))

    data = simbuilder.get_simulation_data()
    ref = simbuilder.controller.get_reference_trajectory()

    analysistools.plot_tracking_error(data, ref, str(savedir))
    analysistools.save_mpc_logs(simbuilder.controller.get_mpc(), str(savedir))
    analysistools.plot_mpc_logs(simbuilder.controller.get_mpc(), str(savedir))

    with open(savedir / "config.json", "w") as config_file:
        config_file.write(json.dumps(config))


def insert_into_path(path: Path, insert: str, index: int = -1) -> Path:
    list_path = list(path.parts)
    new_path_list = list_path[:index] + [insert] + list_path[index:]
    return Path(*new_path_list)
