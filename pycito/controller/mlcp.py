"""
Mixed Linear Complementarity Constraints

Luke Drnach
January 26, 2022
"""
import numpy as np
import warnings 

class PseudoLinearComplementarityConstraint():
    """
    Implements the mixed linear complementarity constraint of the form:
        A*x + c >= 0
        z > = 0
        z * (A*x + c) = 0
    where A is a fixed matrix, c is a fixed vector, x is the free variables, and z are the complementarity variables
    
    PseudoLinearComplementarityConstraint internally adds the slack variable
        s = Ax + c
    and enforces the constraints
        s >= 0
        z >= 0
        s*z = 0
    """
    def __init__(self, A, c):
        # Check dimensions
        assert A.shape[0] == c.shape[0], f"A and c should have the same number of rows"
        self.A = A
        self.c = c

        self.name = 'PseudoLCP'
        self._prog = None
        self._slack = None
        self._xvars = None
        self._zvars = None
        self._lincstr = None

    def __eq__(self, obj):
        """Test for equality of two psuedo-linear complementarity constraints"""
        return type(self) is type(obj) and np.array_equal(self.A, obj.A) and np.array_equal(self.c, obj.c) 

    @classmethod
    def random(cls, xdim, zdim):
        """
        Generates a random psuedo-linear complementarity constraint of the desired size
        
        """
        rng = np.random.default_rng()
        return cls(rng.random((zdim, xdim)), rng.random((zdim,)))

    def str(self):
        return f"{type(self).__name__} with {self.num_free} free variables and {self.dim} complementarity variables"

    def set_description(self, str=None):
        if str:
            self.name = str

    def _add_equality_constraint(self, prog, xvar, svar):
        """
        Imposes the constraint
            A*x + c = s
        Note, the constraint is imposed as
            s - Ax = c
        """
        self._lincstr = prog.AddLinearEqualityConstraint(
                Aeq = np.concatenate([np.eye(self.dim), -self.A], axis=1),
                beq = self.c,
                vars = np.concatenate([svar, xvar], axis=0)
        )
        self._lincstr.evaluator().set_description(f"{self.name}_equality")
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

    def addToProgram(self, prog, xvar, zvar, rvar=None):
        """
        Add the constraint to a mathematical program 
        """
        # Check the dimensions
        assert xvar.shape[0] == self.num_free, f"xvar must be a ({self.num_free}, ) array"
        assert zvar.shape[0] == self.dim, f"zvar must be a ({self.dim},) array"
        # Create and store the slack variables
        self._slack = prog.NewContinuousVariables(rows = self.dim, name=f'{self.name}_slack')
        # Enforce the constraints
        self._prog = prog
        prog = self._add_equality_constraint(prog, xvar, self._slack)
        prog = self._add_nonnegativity_constraint(prog, zvar, self._slack)
        prog = self._add_orthogonality_constraint(prog, zvar, self._slack)
        # Store pointers to the decision variables
        self._xvars = xvar
        self._zvars = zvar
        # Return the program
        return prog

    def initializeSlackVariables(self, xvals = None):
        """
        Set the initial guess for the slack variables
        
        Sets the initial guess to satisfy the constraint for the values of x provided.
        If x is not provided, initializeSlackVariables uses the initial guess for x from the program
        """
        # Get the initial values for x and z
        if self._prog is None:
            return None
        if xvals is None:
            xvals = self._prog.GetInitialGuess(self._xvars)
        # Set the initial value for s
        sval = self.A.dot(xvals) + self.c
        self._prog.SetInitialGuess(self._slack, sval)

    def updateCoefficients(self, A, c):
        """Update the coefficients in the LCP constraint"""
        assert A.shape == self.A.shape, f"The new matrix must have shape {A.shape}"
        assert c.shape == self.c.shape, f"The new vector must have shape {c.shape}"
        self.A = A
        self.c = c
        if self._lincstr is not None:
            self._lincstr.evaluator().UpdateCoefficients(np.concatenate([np.eye(self.dim), -self.A], axis=1),c)

    @property
    def slack(self):
        return self._slack

    @property
    def dim(self):
        return self.A.shape[0]

    @property
    def num_free(self):
        return self.A.shape[1]

    @property
    def cost_weight(self):
        warnings.warn(f"{type(self).__name__} does not have an associated cost")

    @cost_weight.setter
    def cost_weight(self, val):
        warnings.warn(f"{type(self).__name__} does not have an associated cost. The value is ignored.")

class CostRelaxedPseudoLinearComplementarityConstraint(PseudoLinearComplementarityConstraint):
    """
        Recasts the pseudo linear complementarity constraint using an exact penalty method. The constraint is implemented as:
        min a * (s * z)
            s = A*x + c
            s >= 0
            z >= 0
    """
    def __init__(self, A, c):
        super(CostRelaxedPseudoLinearComplementarityConstraint, self).__init__(A, c)
        self._cost_weight = 1
        self._cost = None

    def _add_orthogonality_constraint(self, prog, zvar, svar):
        """
        Add the orthogonality constraint s*z = 0 as a penalty in the cost.
        """
        self._cost = prog.AddQuadraticCost(
                self.cost_matrix, 
                np.zeros((2*self.dim,)),
                np.concatenate([zvar, svar], axis=0) 
        )
        self._cost.evaluator().set_description(f"{self.name}_ProductCost")
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
            if self._cost is not None:
                # Update slack cost
                self._cost.evaluator().UpdateCoefficients(new_Q = self.cost_matrix, new_b = np.zeros((2*self.dim)))
        else:
            raise ValueError("cost_weight must be a nonnegative numeric value")

class VariableRelaxedPseudoLinearComplementarityConstraint(PseudoLinearComplementarityConstraint):
    """
        Recasts the pseudo linear complementarity constraint using an relaxation method. The constraint is implemented as:
        min a * r
            s = A*x + c
            s >= 0
            z >= 0
            r - s*z >= 0
    """
    def __init__(self, A, c):
        super(VariableRelaxedPseudoLinearComplementarityConstraint, self).__init__(A, c)
        self._relax = None
        self._cost = None
        self._cost_weight = np.ones((1,1))

    def addToProgram(self, prog, xvar, zvar, rvar=None):
        """
        Add the constraint to a mathematical program 
        """
        # Check the dimensions
        assert xvar.shape[0] == self.num_free, f"xvar must be a ({self.num_free}, ) array"
        assert zvar.shape[0] == self.dim, f"zvar must be a ({self.dim},) array"
        self._prog = prog
        if rvar is None:
            self._relax = prog.NewContinuousVariables(rows = 1, name = f'{self.name}_relax')
            # If a relaxation variable was not provided, add relaxation constraints
            self._add_relaxation_constraint(prog, self._relax)
        else:
            assert rvar.shape[0] == 1, f"rvar must be a scalar variable"
            self._relax = rvar
        # Create and store the slack variables
        self._slack = prog.NewContinuousVariables(rows = self.dim, name=f'{self.name}_slack')
        # Enforce the constraints
        prog = self._add_equality_constraint(prog, xvar, self._slack)
        prog = self._add_nonnegativity_constraint(prog, zvar, self._slack)
        prog = self._add_orthogonality_constraint(prog, zvar, np.hstack([self._slack, self._relax]))
        # Store pointers to the decision variables
        self._xvars = xvar
        self._zvars = zvar
        # Return the program
        return prog

    def _add_relaxation_constraint(self, prog, relax):
         # Add the slack cost
        self._cost = prog.AddLinearCost(a = self._cost_weight, vars=relax)
        self._cost.evaluator().set_description(f'{self.name}_relax')
        # Add the nonnegativity constraint on the relaxation parameter
        prog.AddBoundingBoxConstraint(np.zeros((1,)), np.full((1,), np.inf), self._relax).evaluator().set_description(f"{self.name}_relax_nonnegativity")
        self.initializeRelaxation()

    def initializeRelaxation(self, val=0.):
        val = np.atleast_1d(val)
        assert val.size == 1, 'relaxation must be a scalar'
        self._prog.SetInitialGuess(self._relax, val)

    def _eval_orthogonality(self, dvals):
        zval, sval, rval = np.split(dvals, [self.dim, 2*self.dim])
        return rval - zval * sval

    def _add_orthogonality_constraint(self, prog, zvar, svar):
        """
        Imposes the constraint
            r - z*s >= 0
        """
        prog.AddConstraint(self._eval_orthogonality, 
            lb = np.zeros((self.dim, )),
            ub = np.full((self.dim, ), np.inf),
            vars = np.concatenate([zvar, svar], axis=0),
            description=f"{self.name}_orthogonality")
        return prog

    @property
    def slack(self):
        return np.concatenate([self._slack, self._relax], axis=0)

    @property
    def cost_weight(self):
        return self._cost_weight

    @cost_weight.setter
    def cost_weight(self, val):
        if isinstance(val, (int, float)):
            assert val >= 0, f"cost_weight must be nonnegative"
            self._cost_weight = np.array([val])
        elif isinstance(val, np.array):
            assert val.size == 1, f"cost_weight must be a scalar"
            self._cost_weight = val
        else:
            raise ValueError("cost_weight must be a nonnegative scalar")
        if self._cost is not None:
            self._cost.evaluator().UpdateCoefficients(self._cost_weight)

class MixedLinearComplementarityConstraint(PseudoLinearComplementarityConstraint):
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
        super(MixedLinearComplementarityConstraint, self).__init__(A, c)
        assert B.shape[0] == B.shape[1], "B must be a square matrix"
        #Store the extra values
        self.B = B
        self.name = 'MLCP'


    def __eq__(self, obj):
        """Two objects are equal if they are of the same type and have the same (A, B, c) parameter values"""
        return type(self) is type(obj) and np.array_equal(self.A, obj.A) and np.array_equal(self.B, obj.B) and np.array_equal(self.c, obj.c) 

    @classmethod
    def random(cls, xdim, zdim):
        rng = np.random.default_rng()
        return cls(rng.random((zdim, xdim)), rng.random((zdim, zdim)), rng.random((zdim,)))

    def addToProgram(self, prog, xvar, zvar):
        """
        Add the constraint to a mathematical program 
        """
        # Check the dimensions
        assert xvar.shape[0] == self.num_free, f"xvar must be a ({self.num_free}, ) array"
        assert zvar.shape[0] == self.dim, f"zvar must be a ({self.dim},) array"
        # Create and store the slack variables
        self._slack = prog.NewContinuousVariables(rows = self.dim, name=f'{self.name}_slack')
        # Enforce the constraints
        self._prog = prog
        prog = self._add_equality_constraint(prog, xvar, zvar, self._slack)
        prog = self._add_nonnegativity_constraint(prog, zvar, self._slack)
        prog = self._add_orthogonality_constraint(prog, zvar, self._slack)
        # Store the variables
        self._xvars = xvar
        self._zvars = zvar
        # Return the program
        return prog

    def initializeSlackVariables(self, xvals=None, zvals=None):
        """
        Initialize the value of the slack variables in the program
        """
        # Get the initial values for x and z
        if self._prog is None:
            return None
        if xvals is None:
            xvals = self._prog.GetInitialGuess(self._xvars)
        if zvals is None:
            zvals = self._prog.GetInitialGuess(self._zvars)
        # Check the dimensions
        assert xvals.shape[0] == self.A.shape[1], f"xvals must be a ({self.num_free}, ) array"
        assert zvals.shape[0] == self.B.shape[1], f"zvals must be a ({self.dim}, ) array"
        # Calculate the slack decision variables
        svals = self.A.dot(xvals) + self.B.dot(zvals) + self.c   
        self._prog.SetInitialGuess(self._slack, svals)     

    def _add_equality_constraint(self, prog, xvar, zvar, svar):
        """
        Imposes the constraint
            A*x + B*z + c = s
        Note, the constraint is imposed as
            s - Ax - Bz = c
        """
        self._lincstr = prog.AddLinearEqualityConstraint(
                Aeq = np.concatenate([np.eye(self.dim), -self.A, -self.B], axis=1),
                beq = self.c,
                vars = np.concatenate([svar, xvar, zvar], axis=0)
        )
        self._lincstr.evaluator().set_description(f"{self.name}_equality")
        return prog

    def updateCoefficients(self, A, B, c):
        """Update the coefficients in the LCP constraint"""
        assert A.shape == self.A.shape, f"The new A matrix must have shape {A.shape}"
        assert B.shape == self.B.shape, f"The new B matrix must have shape {B.shape}"
        assert c.shape == self.c.shape, f"The new c vector must have shape {c.shape}"
        self.A = A
        self.B = B
        self.c = c
        if self._lincstr is not None:
            Aeq = np.concatenate([np.eye(self.dim), -self.A, -self.B], axis=1)
            self._lincstr.evaluator().UpdateCoefficients(Aeq,c)

    @property
    def slack(self):
        return self._slack

class VariableRelaxedMixedLinearComplementarityConstraint(MixedLinearComplementarityConstraint):
    """
        Recasts the mixed linear complementarity constraint using an relaxation method. The constraint is implemented as:
        min a * r
            s = A*x + B*z + c
            s >= 0
            z >= 0
            r - s*z >= 0
    """
    def __init__(self, A, B, c):
        super(VariableRelaxedMixedLinearComplementarityConstraint, self).__init__(A, B, c)
        self._relax = None
        self._cost = None
        self._cost_weight = np.ones((1,1))

    def addToProgram(self, prog, xvar, zvar, rvar=None):
        # Check the dimensions
        assert xvar.shape[0] == self.num_free, f"xvar must be a ({self.num_free}, ) array"
        assert zvar.shape[0] == self.dim, f"zvar must be a ({self.dim},) array"
        self._prog = prog
        # Create and store relaxation variables
        if rvar is None:
            self._relax = prog.NewContinuousVariables(rows=1, name=f'{self.name}_relax')
            self._add_relaxation_constraint(prog, self._relax)
        else:
            assert rvar.shape[0] == 1, f"rvar must be a scalar"
            self._relax = rvar
        # Create and store the slack variables
        self._slack = prog.NewContinuousVariables(rows = self.dim, name=f'{self.name}_slack')
        # Enforce the constraints
        prog = self._add_equality_constraint(prog, xvar, zvar, self._slack)
        prog = self._add_nonnegativity_constraint(prog, zvar, self._slack)
        prog = self._add_orthogonality_constraint(prog, zvar, np.hstack([self._slack, self._relax]))
        # Store the variables
        self._xvars = xvar
        self._zvars = zvar
        # Return the program
        return prog

    def _add_relaxation_constraint(self, prog, relax):
         # Add the slack cost
        self._cost = prog.AddLinearCost(a = self._cost_weight, vars=relax)
        self._cost.evaluator().set_description(f'{self.name}_relax')
        # Add the nonnegativity constraint on the relaxation parameter
        prog.AddBoundingBoxConstraint(np.zeros((1,)), np.full((1,), np.inf), self._relax).evaluator().set_description(f"{self.name}_relax_nonnegativity")
        self.initializeRelaxation()

    def _eval_orthogonality(self, dvals):
        """
        Evaluate the relaxed orthogonality constraint
            r - z*s >= 0
        """
        zval, sval, rval = np.split(dvals, [self.dim, 2*self.dim])
        return rval - zval * sval

    def _add_orthogonality_constraint(self, prog, zvar, svar):
        """
        Imposes the constraint
            r - z*s >= 0
        """
        prog.AddConstraint(self._eval_orthogonality, 
            lb = np.zeros((self.dim, )),
            ub = np.full((self.dim, ), np.inf),
            vars = np.concatenate([zvar, svar], axis=0),
            description=f"{self.name}_orthogonality")
        return prog

    def initializeRelaxation(self, val = 0.):
        """Set the initial guess for the relaxation parameter"""
        val = np.atleast_1d(val)
        assert val.size == 1, "relaxation must be a scalar"
        self._prog.SetInitialGuess(self._relax, val)

    @property
    def slack(self):
        return np.concatenate([self._slack, self._relax], axis=0)

    @property
    def cost_weight(self):
        return self._cost_weight

    @cost_weight.setter
    def cost_weight(self, val):
        if isinstance(val, (int, float)):
            assert val >= 0, f"cost_weight must be nonnegative"
            self._cost_weight = np.array([val])
        elif isinstance(val, np.array):
            assert val.size == 1, f"cost_weight must be a scalar"
            self._cost_weight = val
        else:
            raise ValueError("cost_weight must be a nonnegative scalar")
        if self._cost is not None:
            self._cost.evaluator().UpdateCoefficients(self._cost_weight)

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
        self._cost = None 
                
    def _add_orthogonality_constraint(self, prog, zvar, svar):
        """
        Add the orthogonality constraint s*z = 0 as a penalty in the cost.
        """
        self._cost = prog.AddQuadraticCost(
                self.cost_matrix, 
                np.zeros((2*self.dim,)),
                np.concatenate([zvar, svar], axis=0) 
        )
        self._cost.evaluator().set_description(f"{self.name}_ProductCost")
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
            if self._cost is not None:
                self._cost.evaluator().UpdateCoefficients(new_Q = self.cost_matrix, new_b = np.zeros((2*self.dim)))
        else:
            raise ValueError("cost_weight must be a nonnegative numeric value")

if __name__ == "__main__":
    print("Hello from MLCP")