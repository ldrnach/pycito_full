"""
Mixed Linear Complementarity Constraints

Luke Drnach
January 26, 2022
"""
import numpy as np
import warnings 

class MixedLinearComplementarityConstraint():
    """
    Implements the Mixed Linear Complementarity Constraint of the form:
        A*x + B*z + c >= 0
        z >= 0
        z * (A*x + B*z + c) = 0 
    
    Mixed Linear Complementarity Constraint internally adds the slack variable
        s = A*x + B*z + c
    and enforces the linear complementarity constraint:
        s >= 0
        z >= 0
        s*z = 0
    """    
    def __init__(self, A, B, c):
        # Check the dimensions of the inputs
        assert B.shape[0] == B.shape[1], "B must be a square matrix"
        assert B.shape[1] == c.shape[0], f"c must be a vector with {B.shape[1]} elements"
        assert A.shape[0] == B.shape[1], f"A must have {B.shape[1]} rows"
        # Store the resulting values
        self.A = A
        self.B = B
        self.c = c
        self.name = 'MLCP'
        self._prog = None
        self._var_slack = None


    def str(self):
        return f"{type(self).__name__} with {self.num_free} free variables and {self.dim} complementarity variables"

    def set_description(self, str=None):
        if str:
            self.name = str

    def addToProgram(self, prog, xvar, zvar):
        """
        Add the constraint to a mathematical program 
        """
        # Check the dimensions
        assert self.A.shape[1] == xvar.shape[0], f"xvar must be a ({self.num_free}, ) array"
        assert self.B.shape[1] == zvar.shape[0], f"zvar must be a ({self.dim},) array"
        # Create and store the slack variables
        svar = prog.NewContinuousVariables(rows = self.dim, cols=1, name=f'{self.name}_slack')
        if self._var_slack is None:
            self._var_slack = svar
        else:
            self._var_slack = np.row_stack([self._var_slack, svar])
        # Enforce the constraints
        self._prog = prog
        prog = self._add_equality_constraint(prog, xvar, zvar, svar)
        prog = self._add_nonnegativity_constraint(prog, zvar, svar)
        prog = self._add_orthogonality_constraint(prog, zvar, svar)
        # Return the program
        return prog

    def _add_equality_constraint(self, prog, xvar, zvar, svar):
        """
        Imposes the constraint
            A*x + B*z + c = s
        """
        cstr = prog.AddLinearEqualityConstraint(
            Aeq = np.concatenate([np.eye(self.dim), -self.A, -self.B], axis=1),
            beq = self.c,
            vars = np.concatenate([svar, xvar, zvar], axis=0)
        )
        cstr.evaluator().set_description(f"{self.name}_equality")
        return prog

    def _add_nonnegativity_constraint(self, prog, zvar, svar):
        """
        Imposes the constraints
            z >= 0
            s >= 0 
        """
        prog.AddBoundingBoxConstraint(
            np.zeros((2*self.dim, )),
            np.full((2*self.dim, ), np.inf), 
            np.concatenate([zvar, svar], axis=0)
        ).evaluator().set_description(f"{self.name}_nonnegativity")
        return prog

    def _add_orthogonality_constraint(self, prog, zvar, svar):
        """
        Imposes the constraint
            z*s = 0
        """
        prog.AddConstraint(self._eval_orthogonality, 
            lb = np.zeros((self.dim, )),
            ub = np.zeros((self.dim, )),
            vars = np.concatenate([zvar, svar], axis=0),
            description=f"{self.name}_orthogonality")
        return prog
    
    def _eval_orthogonality(self, dval):
        """
        evaluates the constraint
            z * s 
        """
        zval, sval = np.split(dval, 2)
        return sval * zval

    @property
    def dim(self):
        return self.B.shape[1]

    @property
    def num_free(self):
        return self.A.shape[1]

    @property
    def cost_weight(self):
        warnings.warn(f"{type(self).__name__} does not have an associated cost")

    @cost_weight.setter
    def cost_weight(self, val):
        warnings.warn(f"{type(self).__name__} does not have an associated cost. The value is ignored.")

    @property
    def var_slack(self):
        return self._var_slack

class CostRelaxedMixedLinearComplementarity(MixedLinearComplementarityConstraint):
    """
    Recasts the mixed linear complementarity constraint using an exact penalty method. The constraint is implemented as:
        min a * (s * z)
            s = A*x + B*z + c
            s >= 0
            z >= 0
    """
    def __init__(self, A, B, c):
        super(CostRelaxedMixedLinearComplementarity, self).__init__(A, B, c)
        self._cost_weight = 1
        self._costs = [] 
        
    def _add_orthogonality_constraint(self, prog, zvar, svar):
        """
        Add the orthogonality constraint s*z = 0 as a penalty in the cost.
        """
        cost = prog.AddQuadraticCost(
                self.cost_matrix, 
                np.zeros((2*self.dim,)),
                np.concatenate([zvar, svar], axis=0) 
        )
        cost.evaluator().set_description(f"{self.name}_ProductCost")
        self._costs.append(cost)
        return prog

    @property
    def cost_matrix(self):
        w = self._cost_weight * np.ones((self.dim,))
        Q = np.diag(w, k=self.dim)
        return Q + Q.T

    @property
    def cost_weight(self):
        return self._cost_weight

    @cost_weight.setter
    def cost_weight(self, val):
        if (type(val) == int or type(val) == float) and val >= 0.:
            #Store the cost weight value
            self._cost_weight = val
            # Update slack cost
            for cost in self._costs:
                cost.evaluator().UpdateCoefficients(new_Q = self.cost_matrix, new_b = np.zeros((2*self.dim)))
        else:
            raise ValueError("cost_weight must be a nonnegative numeric value")

if __name__ == "__main__":
    print("Hello from MLCP")