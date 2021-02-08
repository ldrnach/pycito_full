import os
from sys import exit
from pydrake.autodiffutils import AutoDiffXd
import pickle
import numpy as np

SNOPT_DECODER = {
    0: "finished successfully",
    1: "optimality conditions satisfied",
    2: "feasible point found",
    3: "requested accuracy could not be achieved",
    11: "infeasible linear constraints",
    12: "infeasible linear equalities",
    13: "nonlinear infeasibilities minimized",
    14: "infeasibilities minimized",
    21: "unbounded objective",
    22: "constraint violation limit reached",
    31: "iteration limit reached",
    32: "major iteration limit reached",
    33: "the superbasics limit is too small",
    41: "current point cannot be improved",
    42: "singular basis",
    43: "cannot satisfy the general constraints",
    44: "ill-conditioned null-space basis",
    51: "incoorrect objective derivatives",
    52: "incorrect constraint derivatives",
    61: "undefined function at the first feasible point",
    62: "undefined function at the initial point",
    63: "unable to proceed in undefined region",
    71: "terminated during function evaluation",
    72: "terminated during constraint evaluation",
    73: "terminated during objective evaluation",
    74: "termianted from monitor routine",
    81: "work arrays must have at least 500 elements",
    82: "not enough character storage",
    83: "not enough integer storage",
    84: "not enough real storage",
    91: "invalid input argument",
    92: "basis file dimensions do not match this problem",
    141: "wrong number of basic variables",
    142: "error in basis package"
}

def save(filename, data):
    """ pickle data in the specified filename """
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(filename, "wb") as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

def load(filename):
    """ unpickle the data in the specified filename """
    with open(filename, "rb") as input:
        data = pickle.load(input)
    return data

def FindResource(filename):
    if not os.path.isfile(filename):
        exit(f"{filename} not found")
    else:
        return os.path.abspath(filename)
    
def CheckProgram(prog):
    """
    Return true if the outputs of all costs and constraints in MathematicalProgram are valid
    
    Arguments:
        prog: a MathematicalProgram pyDrake object
    """
    status = True
    # Check that the outputs of the costs are all scalars
    for cost in prog.generic_costs():
        # Evaluate the cost with floats
        try:
            xs = [1.]*len(cost.variables())
            cost.evaluator().Eval(xs)
        except RuntimeError as err:
            status = False
            print(f"Evaluating {cost.evaluator().get_description()} with floats produces a RuntimeError")
        # Evaluate with AutoDiff arrays
        try:
            xd = [AutoDiffXd(1.)] * len(cost.variables())
            cost.evaluator().Eval(xd)
        except RuntimeError as err:
            status = False
            print(f"Evaluating {cost.evaluator().get_description()} with AutoDiffs produces a RuntimeError")
    # Check that the outputs of all constraints are vectors
    for cstr in prog.generic_constraints():
        # Evaluate the constraint with floats
        try:
            xs = [1.]*len(cstr.variables())
            cstr.evaluator().Eval(xs)
        except RuntimeError as err:
            status = False
            print(f"Evaluating {cstr.evaluator().get_description()} with floats produces a RuntimeError")
        except ValueError as err:
            status = False
            print(f"Evaluating {cstr.evaluator().get_description()} with floats resulted in a ValueError")
        # Evaluate constraint with AutoDiffXd
        try:
            xd = [AutoDiffXd(1.)] * len(cstr.variables())
            cstr.evaluator().Eval(xd)
        except RuntimeError as err:
            status = False
            print(f"Evaluating {cstr.evaluator().get_description()} with AutoDiffs produces a RuntimeError")
        except ValueError as err:
            print(f"Evaluating {cstr.evaluator().get_description()} with AutoDiffs produces a ValueError")
    # Return the status flag
    return status

def GetKnotsFromTrajectory(trajectory):
    breaks = trajectory.get_segment_times()
    values = trajectory.vector_values(breaks)
    return (breaks, values)

def printProgramReport(result, prog=None, filename=None):
    """print out information about the result of the mathematical program """
    # Print out general information
    report = f"Optimization successful? {result.is_success()}\n"
    report += f"Optimal cost = {result.get_optimal_cost()}\n"
    report += f"Solved with {result.get_solver_id().name()}\n"
    # Print out SNOPT specific information
    if result.get_solver_id().name() == "SNOPT/fortran":
        exit_code = result.get_solver_details().info
        report += f"SNOPT Exit Status {exit_code}: {SNOPT_DECODER[exit_code]}\n"
        if prog is not None:
            # Filter out the empty infeasible constraints
            infeasibles = result.GetInfeasibleConstraintNames(prog)
            infeas = [name.split("[")[0] for name in infeasibles]
            report += f"Infeasible constriants: {set(infeas)}\n"
    if filename is None:
        print(report)
    else:
        with open(filename, "w") as file:
            file.write(report)

def quat2rpy(quat):
    """
    Convert a quaternion to Roll-Pitch-Yaw
    
    Arguments:
        quaternion: a (4,n) numpy array of quaternions
    
    Return values:
        rpy: a (3,n) numpy array of roll-pitch-yaw values
    """
    rpy = np.zeros((3, quat.shape[1]))
    rpy[0,:] = np.arctan2(2*(quat[0,:]*quat[1,:] + quat[2,:]*quat[3,:]),
                         1-2*(quat[1,:]**2 + quat[2,:]**2))
    rpy[1,:] = np.arcsin(2*(quat[0,:]*quat[2,:]-quat[3,:]*quat[1,:]))
    rpy[2,:] = np.arctan2(2*(quat[0,:]*quat[3,:]+quat[1,:]*quat[2,:]),
                        1-2*(quat[2,:]**2 + quat[3,:]**2))
    return rpy

def plot_complementarity(ax, y1, y2, label1, label2):
    """
        Plots two traces in the same axes using different y-axes. Aligns the y-axes at zero

        Arguments:
            ax: The axis on which to plot
            y1: The first sequence to plot
            y2: The second sequence to plot
            label1: The y-axis label for the first sequence, y1
            label2: The y-axis label for the second sequence, y2
    """
    x = range(0, len(y1))
    color = "tab:red"
    ax.set_ylabel(label1, color = color)
    ax.plot(x, y1, color=color, linewidth=1.5)
    # Create the second axis 
    ax2 = ax.twinx()
    color = "tab:blue"
    ax2.set_ylabel(label2, color=color)
    ax2.plot(x, y2, color=color, linewidth=1.5)
    # Align the axes at zero
    align_axes(ax,ax2)

def align_axes(ax, ax2):
    """
        For a plot with two y-axes, aligns the two y-axes at 0

        Arguments:
            ax: Reference to the first of the two y-axes
            ax2: Reference to the second of the two y-axes
    """
    lims = np.array([ax.get_ylim(), ax2.get_ylim()])
    # Pad the limits to make sure there is some range
    lims += np.array([[-1,1],[-1,1]])
    lim_range = lims[:,1] - lims[:,0]
    lim_frac = lims.transpose() / lim_range
    lim_frac = lim_frac.transpose()
    new_frac = np.array([min(lim_frac[:,0]), max(lim_frac[:,1])])
    ax.set_ylim(lim_range[0]*new_frac)
    ax2.set_ylim(lim_range[1]*new_frac)