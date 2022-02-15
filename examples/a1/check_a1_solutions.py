"""
check_a1_solutions: tools for checking the solutions to a1 trajectory optimization
Luke Drnach
Feburary 15, 2022
"""
import numpy as np
import os

import pycito.trajopt.contactimplicit as ci
from pycito.utilities import load, FindResource, MathProgIterationPrinter
from pycito.systems.A1.a1 import A1VirtualBase

from examples.a1.a1trajopttools import make_a1_trajopt_linearcost

SOURCE = os.path.join('examples','a1','foot_tracking_gait','twostepopt','weight_1e+03')
FILENAME = 'trajoptresults.pkl'

def check_solution():
    data = load(FindResource(os.path.join(SOURCE, FILENAME)))
    N = data['time'].size
    T = data['time'][-1]
    a1 = A1VirtualBase()
    a1.Finalize()
    trajopt = make_a1_trajopt_linearcost(a1, N, [T, T])
    viewer = ci.ContactConstraintViewer(trajopt, data)
    cstrvals = viewer.calc_constraint_values()
    #viewer.plot_dynamic_defects(data['time'][:], cstrvals['dynamics'][: , :], show=True, savename=None)
    printer = MathProgIterationPrinter(trajopt.prog)
    cstrviol = printer.calc_constraints(viewer.all_vals)
    print("Hello")


if __name__ == '__main__':
    check_solution()