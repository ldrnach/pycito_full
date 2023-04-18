from dataclasses import dataclass
from typing import Dict, Literal, Optional, Union


@dataclass
class SNOPTConfig:
    major_feasibility_tolerance: Optional[float] = None
    major_optimality_tolerance: Optional[float] = None
    scale_option: Optional[Literal[0, 1, 2]] = None
    major_step_limit: Optional[float] = None
    superbasics_limit: Optional[float] = None
    linesearch_tolerance: Optional[float] = None
    iterations_limit: Optional[int] = None
    use_basis_file: bool = False

    @property
    def solver_options(self) -> Dict[str, Union[float, int]]:
        options = {
            "Major feasibility tolerance": self.major_feasibility_tolerance,
            "Major optimality tolerance": self.major_optimality_tolerance,
            "Scale option": self.scale_option,
            "Major step limit": self.major_step_limit,
            "Superbasics limit": self.superbasics_limit,
            "Linesearch tolerance": self.linesearch_tolerance,
            "Iterations limit": self.iterations_limit,
        }
        for key, value in options.items():
            if value is None:
                options.pop(key)
        return options
