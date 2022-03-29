import numpy as np
from pydrake.all import MathematicalProgram, SnoptSolver, OsqpSolver, GurobiSolver, IpoptSolver, ChooseBestSolver, MakeSolver
from collections import defaultdict
import re

class OptimizationMixin():
    def __init__(self):
        super().__init__()
        self._prog = MathematicalProgram()
        self._solver = None
        self.solver_options = {}

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
        return self.solver.Solve(self.prog)

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
        vars = self.get_decision_variable_dicionary()
        soln = {}
        for key, value in vars.items():
            soln[key] = result.GetSolution(value)
        # Store the final, optimal cost
        soln['total_cost'] = result.get_optimal_cost()
        # Store solution meta-data
        soln['success'] = result.is_success()
        soln['solver'] = result.solver_id().name()
        if soln['solver'] == 'SNOPT/fortran':
            soln['exitcode'] = result.solver_details().info
        elif id == 'OSQP':
            soln['exitcode'] = result.get_solver_details().status_val
        elif id =='IPOPT':
            soln['exitcode'] = result.get_solver_details().status
        elif id =='Gurobi':
            soln['exitcode'] =  result.get_solver_details().optimization_status
        else:
            soln['exitcode'] = np.NaN
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

if __name__ == '__main__':
    print('Hello from optimization!')