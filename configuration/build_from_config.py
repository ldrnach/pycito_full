from types import ModuleType
from typing import Any, Protocol


class ClassConfiguration(Protocol):
    @property
    def type(self) -> str:
        pass


def build_from_config(
    module: ModuleType, config: ClassConfiguration, **kwargs: dict
) -> Any:
    classtype = getattr(module, config.type)
    return classtype.build_from_config(config, **kwargs)
