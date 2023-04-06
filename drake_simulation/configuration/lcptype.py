from typing import Literal

LCP = Literal[
    "PseudoLinearComplementarityConstraint",
    "CostRelaxedPseudoLinearComplementarityConstraint",
    "VariableRelaxedPseudoLinearComplementarityConstraint",
    "VariableRelaxedAugmentedPseudoLinearComplementarityConstraint",
    "ConstantRelaxedPseudoLinearComplementarityConstraint",
    "CentralPathPseudoLinearComplementarityConstraint",
    "MixedLinearComplementarityConstraint",
    "VariableRelaxedMixedLinearComplementarityConstraint",
    "CostRelaxedMixedLinearComplementarity",
]
