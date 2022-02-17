"""
Tools for modeling contact semiparametrically

Luke Drnach
February 16, 2022
"""

import numpy as np

class SemiparametricModel():
    def __init__(self, prior, kernel):
        self.prior = prior
        self.kernel = kernel
        self._kernel_matrix = None
        self._kernel_weights = None


class SemiparametricContactModel()    