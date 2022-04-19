import numpy as np
from collections import defaultdict
import re, time
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from pydrake.all import MathematicalProgram, SnoptSolver, OsqpSolver, GurobiSolver, IpoptSolver, ChooseBestSolver, MakeSolver

import pycito.utilities as utils
import pycito.decorators as deco

class OptimizationMixin():
    def __init__(self):
        super().__init__()
        self._prog = MathematicalProgram()
        self._solver = None
        self.solver_options = {}
        self._logger = OptimizationLogger(self)
        self._log_enabled = False

    def enableLogging(self):
        self._log_enabled = True

    def disableLogging(self):
        self._log_enabled = False

    def useIpoptSolver(self):
        self.solver = IpoptSolver()

    def useSnoptSolver(self):
        self.solver = SnoptSolver()

    def useGurobiSolver(self):
        self.solver = GurobiSolver()

    def useOsqpSolver(self):
        self.solver = OsqpSolver()

    def setSolverOptions(self, options_dict):
        for key, value in options_dict.items():
            self.solver_options[key] = value

    def solve(self):
        """Solve the optimization problem"""
        if self._solver is None:
            self.solver = MakeSolver(ChooseBestSolver(self._prog))
        assert self.solver.AreProgramAttributesSatisfied(self.prog), f'The costs and constraints do not satisfy the requirements for using {self.solver.solver_id().name()} as a solver. Please choose a different solver'
        # Set the solver options
        for key, value in self.solver_options.items():
            self.prog.SetSolverOption(self.solver.solver_id(), key, value)
        # Solve and return the solution
        start = time.perf_counter()
        result =  self.solver.Solve(self.prog)
        elapsed = time.perf_counter() - start        
        if self._log_enabled:
            self._logger.log(result, elapsed)

        return result

    def get_decision_variable_dictionary(self):
        """
            Returns the decision variables in a dictionary, organized by variable name
            This version assumes the variables were added only in vectors (with rows, but not with columns)
        """
        dvars = self.prog.decision_variables()
        named_vars = defaultdict(list)
        # Organize repeating variables
        for dvar in dvars:
            named_vars[dvar.get_name()].append(dvar)
        # Organize variables with the same name, but added in blocks
        names = defaultdict(list)
        for name in named_vars.keys():
            name_parts = re.split('\(|\)', name)
            names[name_parts[0]].append((name, int(name_parts[1])))
        # Make the final output array
        var_dict = defaultdict(list)
        maxidx = lambda x: max([t[1] for t in x])
        for name, values in names.items():
            var_dict[name] = [None] * (maxidx(values)  + 1)
            for value in values:
                var_dict[name][value[1]] = named_vars[value[0]]
        # Finally, convert to arrays
        for key, value in var_dict.items():
            var_dict[key] = np.asarray(value)
        return var_dict

    def result_to_dict(self, result):
        """
            Store the data in MathematicalProgramResult in a dictionary
        """
        # Store the final values of the decision variables
        vars = self.get_decision_variable_dictionary()
        soln = {}
        for key, value in vars.items():
            soln[key] = result.GetSolution(value)
        # Store the final, optimal cost
        soln['total_cost'] = result.get_optimal_cost()
        # Add cost and constraint meta-data
        soln['costs'] = self.get_costs(result)
        soln['constraints'] = self.get_constraints(result)
        # Store solution meta-data
        soln['success'] = result.is_success()
        soln['solver'] = result.get_solver_id().name()
        if soln['solver'] == 'SNOPT/fortran':
            soln['exitcode'] = result.get_solver_details().info
        elif id == 'OSQP':
            soln['exitcode'] = result.get_solver_details().status_val
        elif id =='IPOPT':
            soln['exitcode'] = result.get_solver_details().status
        elif id =='Gurobi':
            soln['exitcode'] =  result.get_solver_details().optimization_status
        else:
            soln['exitcode'] = 0
        return soln

    def get_costs(self, result=None):
        """Get all the cost function values"""
        costs = defaultdict(int)
        if result is None:
            eval_vars = self.prog.GetInitialGuess
        else:
            eval_vars = result.GetSolution

        for cost in self.prog.GetAllCosts():
            dvars = cost.variables()
            val = cost.evaluator().Eval(eval_vars(dvars))
            name = cost.evaluator().get_description()
            costs[name] += val
        return costs

    def get_constraints(self, result=None):
        """Get all the constraint violations"""
        cstrs = defaultdict(int)
        if result is None:
            eval_vars = self.prog.GetInitialGuess
        else:
            eval_vars = result.GetSolution
        for cstr in self.prog.GetAllConstraints():
            name = cstr.evaluator().get_description()
            # Constraint violation
            val = cstr.evaluator().Eval(eval_vars(cstr.variables()))
            lb = cstr.evaluator().lower_bound()
            ub = cstr.evaluator().upper_bound()
            lb_viol = np.minimum(val - lb, np.zeros(lb.shape))
            lb_viol[np.isinf(lb)] = 0.
            ub_viol = np.maximum(val - ub, np.zeros(ub.shape))
            ub_viol[np.isinf(ub)] = 0.
            viol = sum(abs(lb_viol)) + sum(abs(ub_viol))
            cstrs[name] += viol
        return cstrs

    def initial_guess_dictionary(self):
        """Create a results dictionary based on the initial guess"""
        vars = self.get_decision_variable_dictionary()
        soln = {}
        for key, value in vars.items():
            soln[key] = self.prog.GetInitialGuess(value)
        soln['costs'] = self.get_costs()
        soln['constraints'] = self.get_constraints()
        return soln

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        assert solver.available(), f"{solver.solver_id().name()} is not currently available for use as a solver. Please choose a different solver."
        assert solver.enabled(), f"{solver.solver_id().name()} is not enabled for use at runtime. Please choose a different solver"
        self._solver = solver
        self.solver_options = {}

    @property
    def prog(self):
        return self._prog

    @property
    def logger(self):
        return self._logger

class OptimizationLogger():
    def __init__(self, problem):
        assert issubclass(type(problem), OptimizationMixin), 'problem must inherit from OptimizationMixin'
        self.problem = problem
        self.logs = []
        self.guess_logs = []

    def save(self, filename):
        """
            Save the logs to a file
            NOTE: Save only saves the logs, not the pointer to the original optimization problem
        """
        utils.save(filename, (self.logs, self.guess_logs))

    @classmethod
    def load(self, filename):
        """
            Load the logs from a file
            NOTE: load does not restore the original optimization problem, only the result logs. In it's place, we put an empty OptimizationMixin object
        """
        logger = OptimizationLogger(OptimizationMixin())
        data = utils.load(utils.FindResource(filename))
        logger.logs, logger.guess_logs = data[0], data[1]
        return logger

    def log(self, results, elapsed=None):
        """Log the results of calling the optimization program"""
        self.logs.append(self.problem.result_to_dict(results))
        self.guess_logs.append(self.problem.initial_guess_dictionary())
        if elapsed is not None:
            self.logs[-1]['solvetime'] = elapsed

    def plot(self, show = False, savename=None):
        """
            Plots the exit code, costs, and constraints of the logged optimization problems
        """
        fig, axs = self.plot_status(show=False, savename=utils.append_filename(savename, 'Status'))
        fig1, axs1 = self.plot_costs(show=False, savename=utils.append_filename(savename,'Costs'))
        fig2, axs2 = self.plot_constraints(show=False, savename=utils.append_filename(savename, 'Constraints'))
        if show:
            plt.show()
        return [fig, fig1, fig2], [axs, axs1, axs2]

    @deco.showable_fig
    @deco.saveable_fig
    def plot_status(self, axs=None):
        """
            Plot the exit status code on the given axis
        """
        if axs is None:
            fig, axs = plt.subplots(1,1)
        else:
            plt.sca(axs)
            fig = plt.gcf()
        code = [log['exitcode'] for log in self.logs]
        x = np.arange(len(code))
        axs.scatter(x, code)
        axs.set_ylabel('Exit Code')
        axs.set_xlabel('Problem number')
        return fig, axs

    @deco.showable_fig
    @deco.saveable_fig
    def plot_constraints(self, axs=None):
        """
            Plot the constraint values on the given axis
        """
        if axs is None:
            fig, axs = plt.subplots(1,1)
        else:
            plt.sca(axs)
            fig = plt.gcf()
        for key, value in self.constraint_log_array().items():
            x = np.arange(value.shape[0])
            axs.plot(x, value, linewidth=1.5, label=key)
        axs.set_ylabel('Violation')
        axs.set_xlabel('Problem Number')
        axs.set_yscale('symlog', linthresh=1e-6)
        axs.grid(True)
        axs.legend()
        axs.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
        return fig, axs

    @deco.showable_fig
    @deco.saveable_fig
    def plot_costs(self, axs=None):
        """Plot the cost values on the given axis"""
        if axs is None:
            fig, axs = plt.subplots(1,1)
        else:
            plt.sca(axs)
            fig = plt.gcf()
        for key, value in self.cost_log_array().items():
            x = np.arange(value.shape[0])
            axs.plot(x, value, linewidth=1.5, label=key)
        axs.set_ylabel('Cost')
        axs.set_xlabel('Problem Number')
        axs.set_yscale('symlog', linthresh=1e-6)
        axs.grid(True)
        axs.legend()
        return fig, axs

    def cost_log_array(self):
        """Return all the costs as a dictionary of arrays"""
        costs = {key: [log['costs'][key] for log in self.logs] for key in self.logs[0]['costs'].keys()}
        for key in costs.keys():
            costs[key] = np.asarray(costs[key])
        return costs

    def constraint_log_array(self):
        """Return all the constraint violations as a dictionary of arrays"""
        cstrs = {key: [log['constraints'][key] for log in self.logs] for key in self.logs[0]['constraints'].keys()}
        for key in cstrs.keys():
            cstrs[key] = np.asarray(cstrs[key])
        return cstrs

if __name__ == '__main__':
    print('Hello from optimization!')