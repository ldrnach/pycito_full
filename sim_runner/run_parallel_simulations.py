import concurrent.futures
import os
from pathlib import Path
from typing import Iterator, List

from configuration.load_simulator_config import load_simulator_config
from sim_runner.run_simulation import run_simulation

CONFIG_DIR = Path("config", "runs")


def run_parallel_simulations() -> None:
    # Get the configurations
    paths = get_configurations()
    if len(paths) == 0:
        print("Found no configurations")
        return
    # Load them
    configs = [load_simulator_config(path) for path in paths]
    directories = [path.with_suffix("") for path in paths]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(run_simulation, configs, directories)


def get_configurations() -> List[Path]:
    path_configs: List[Path] = []
    for path, _, files in os.walk(CONFIG_DIR):
        for file in files:
            if file.endswith(".json"):
                path_configs.append(Path(path) / file)
    return path_configs


if __name__ == "__main__":
    run_parallel_simulations()
