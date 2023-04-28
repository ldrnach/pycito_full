import json
from pathlib import Path

from dacite import from_dict

from configuration.simulator import DrakeSimulatorConfig


class NotJsonError(Exception):
    pass


def load_simulator_config(filename: Path) -> DrakeSimulatorConfig:
    # Check if the file exists
    if not filename.exists():
        raise FileNotFoundError(f"{filename} does not exist")
    if filename.suffix != ".json":
        raise NotJsonError(f"{filename} is not a json file")

    with open(filename) as file:
        data = json.load(file)
    return from_dict(data_class=DrakeSimulatorConfig, data=data)
