from dataclasses import dataclass, asdict
from typing import Literal, Optional, Dict, Union


@dataclass
class SNOPTConfig:
    major_feasibility_tolerance: Optional[float] = None
    major_optimality_tolerance: Optional[float] = None
    scale_option: Optional[Literal[0, 1, 2]] = None
    superbasics_limit: Optional[float] = None
    linesearch_tolerance: Optional[float] = None
    iterations_limit: Optional[int] = None
    use_basis_file: bool = False

    @property
    def solver_options(self) -> Dict[str, Union[float, int]]:
        options = asdict(self)
        for key, value in options.items():
            if value is None:
                options.pop(key)

        options.pop("use_basis_file")
        return options
