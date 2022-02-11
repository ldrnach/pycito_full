"""
    Tools for running speed tests on model predictive control

    Luke Drnach
    February 11, 2022
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from pydrake.all import OsqpSolver, SnoptSolver

import pycito.controller.mpc as mpc
import pycito.utilities as utils
import pycito.decorators as deco

def get_constraint_violation(prog, result):
    violation = 0
    for cstr in prog.GetAllConstraints():
        dvals = result.GetSolution(cstr.variables())
        # Evaluate the constraint
        cvals = cstr.evaluator().Eval(dvals)
        lb = cstr.evaluator().lower_bound()
        ub = cstr.evaluator().upper_bound()
        # Calculate total constraint bound violation
        lb_viol = np.minimum(cvals - lb, np.zeros_like(lb))
        lb_viol[np.isinf(lb)] = 0
        ub_viol = np.maximum(cvals - ub, np.zeros_like(ub))
        ub_viol[np.isinf(ub)] = 0
        violation += sum(abs(lb_viol)) + sum(abs(ub_viol))
    return violation

class SpeedTestResult():
    def __init__(self, max_horizon, num_samples):
        # Create arrays to store the speedtest data
        self.startup_times = np.zeros((num_samples, max_horizon))
        self.solve_times = np.zeros((num_samples, max_horizon))
        self.successful_solves = np.zeros((num_samples, max_horizon), dtype=bool)
        self.cost_vals = np.zeros((num_samples, max_horizon))
        self.cstr_vals = np.zeros((num_samples, max_horizon))

    @staticmethod
    def load(filename):
        data = utils.load(utils.FindResource(filename))
        if isinstance(data, SpeedTestResult):
            return data
        else:
            raise ValueError(f"{filename} does not contain a SpeedTestResult")

    @staticmethod
    def _plot_statistics(axs, data, filter):
        xpoints = np.arange(1, data.shape[1]+1)
        # Calculate the mean and standard deviation of the data
        datacopy = data.copy()
        datacopy[~filter] = np.nan
        average = np.nanmean(datacopy, axis=0)
        deviation = np.nanstd(datacopy, axis=0)
        # Now plot the mean and standard deviation

        axs.fill_between(xpoints, average-deviation, average+deviation, alpha=0.5)
        axs.plot(xpoints, average)
        return axs

    def save(self, filename):
        utils.save(filename, self)

    def record_times(self, startup, solve, sample_number, horizon ):
        """Record the startup and solve times for the speedtest"""
        self.startup_times[sample_number, horizon-1] = startup
        self.solve_times[sample_number, horizon-1] = solve

    def record_results(self, success, cost, cstr, sample_number, horizon):
        """Record the MPC results for the speedtest"""
        self.successful_solves[sample_number, horizon-1] = success
        self.cost_vals[sample_number, horizon-1] = cost
        self.cstr_vals[sample_number, horizon-1] = cstr

    def plot(self, show=False, savename=None):
        self.plot_times(show=show, savename=utils.append_filename(savename, 'times'))
        self.plot_results(show=show, savename=utils.append_filename(savename,'solverresults'))
        
    @deco.showable_fig
    @deco.saveable_fig
    def plot_times(self):
        """Plot the creation and solve times of the speedtest"""
        fig, axs = plt.subplots(2,1)
        self._plot_statistics(axs[0], self.startup_times, self.successful_solves)
        self._plot_statistics(axs[1], self.solve_times, self.successful_solves)
        axs[0].set_ylabel('MPC Creation time (s)')
        axs[1].set_ylabel('MPC Solve time (s)')
        axs[1].set_xlabel('MPC Horizon (samples)')
        return fig, axs

    @deco.showable_fig
    @deco.saveable_fig
    def plot_results(self):
        fig, axs = plt.subplots(3, 1)
        # Plot the total number of successful solves
        xrange = np.arange(1, self.successful_solves.shape[1]+1)
        successes = np.sum(self.successful_solves, axis=0)
        axs[0].plot(xrange, successes, linewidth=1.5)
        # Plot the costs and constraints
        self._plot_statistics(axs[1], self.cost_vals, self.successful_solves)
        self._plot_statistics(axs[2], self.cstr_vals, self.successful_solves)
        axs[0].set_ylabel('Number successful solves')
        axs[1].set_ylabel('Total cost')
        axs[2].set_ylabel('Total constraint violation')
        axs[2].set_xlabel('MPC Horizon (samples)')
        return fig, axs


class MPCSpeedTest():
    def __init__(self, reftraj, max_horizon=None):
        self.reftraj = reftraj
        if max_horizon is None:
            self.max_horizon = self.reftraj.num_timesteps
        else:
            self.max_horizon = max_horizon

        self.solver = OsqpSolver()
        self.solveroptions = {'eps_abs': 1e-6,
                            'eps_rel': 1e-6}
        
    def useOsqpSolver(self):
        self.solver = OsqpSolver()
        self.solveroptions = {'eps_abs':1e-6,
                            'eps_rel':1e-6}

    def useSnoptSolver(self):
        self.solver = SnoptSolver()
        self.solveroptions = {'Major feasibility tolerance': 1e-6,
                            'Major optimality tolerance': 1e-6}

    def run_speedtests(self, state_samples):
        speedResult = SpeedTestResult(self.max_horizon, state_samples.shape[1])

        for horizon in range(1, self.max_horizon+1):
            print(f'Running MPC speed tests for horizon {horizon}: ')    
            controller = mpc.LinearContactMPC(self.reftraj, horizon)
            controller._solver = self.solver
            controller.setSolverOptions(self.solveroptions)
            for sampleidx, x_sample in enumerate(state_samples.T):
                # Create MPC
                start = time.perf_counter()
                controller.create_mpc_program(0., x_sample)
                create_time = time.perf_counter() - start
                # Solve MPC
                start = time.perf_counter()
                result = controller.solve()
                solve_time = time.perf_counter() - start
                # Record results
                speedResult.record_times(create_time, solve_time, sampleidx, horizon)
                cstr = get_constraint_violation(controller.prog, result)
                speedResult.record_results(result.is_success(), result.get_optimal_cost(), cstr, sampleidx, horizon)
            print(f"\t{sum(speedResult.successful_solves[:, horizon-1])} of {state_samples.shape[1]} solved successfully")
        
        return speedResult