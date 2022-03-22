from pydrake.all import MathematicalProgram, SnoptSolver, OsqpSolver, GurobiSolver, IpoptSolver, ChooseBestSolver, MakeSolver

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