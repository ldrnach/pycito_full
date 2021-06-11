"""
complementarity: implementations of complementarity constraints for use in optimization problems.

Luke Drnach
June 11, 2021
"""
import numpy as np
from abc import ABC, abstractmethod
import warnings

class ComplementarityFunction(ABC):
    """
    Base class for implementing complementarity constraint functions of the form:
        f(x) >= 0
        z >= 0
        f(x) * z <= s
    where s is a slack variable and s=0 enforces strict complementarity
    """
    def __init__(self, fcn, xdim=1, zdim=1):
        """
            Creates the complementarity constraint function. Stores a reference to the nonlinear function, the dimensions of the x and z variables, and sets the slack to 0
        """
        self.fcn = fcn
        self.xdim = xdim
        self.zdim = zdim
        self.slack = 0.
        self.name = self.fcn.__name__
    
    def __call__(self, vars):
        """Evaluate the constraint"""
        return self.eval(vars)

    @abstractmethod
    def eval(self, vars):
        """Concrete implementation of the evaluator called by __call__"""

    @abstractmethod
    def lower_bound(self):
        """Returns the lower bound of the constraint"""

    @abstractmethod
    def upper_bound(self):
        """Returns the upper bound of the constraint"""

    def set_description(self, name=None):
        """Set a string description for the constraint"""
        if name is None:
            self.name = self.fcn.__name__
        else:
            self.name = name

    def eval_product(self, vars):
        """Return the product of the complementarity function and variables"""
        x, z  = np.split(vars, [self.xdim])
        return z*self.fcn(x)

    def addToProgram(self, prog, xvars, zvars):
        """Add the complementarity constraint to a program"""
        self._check_vars(xvars, zvars)
        # Add the linear nonnegativity constraint
        self._addNonnegativityConstraint(prog, zvars)
        # Add the non-linear constraints
        for n in range(xvars.shape[1]):
            prog.AddConstraint(self.eval_product, lb =  )




    def _addNonnegativityConstraint(self, prog, zvars):
        """Add the linear nonnegativity constraint on the complementarity variables"""
        cstr = prog.AddBoundingBoxConstraint(lb = 0., ub = np.inf, vars = zvars.flatten())
        cstr.evaluator().set_description(self.name + "_nonnegativity")

    def _check_vars(self, xvars, zvars):
        """ Check that the decision variables are appropriately sized"""
        nX, N = xvars.shape
        nZ, M = zvars.shape
        if nX != self.xdim:
            raise ValueError(f"Expected {self.xdim} xvars but got {nX} instead")
        if nZ != self.zdim:
            raise ValueError(f"Expected {self.zdim} zvars but got {nZ} instead")
        if N != M:
            raise ValueError(f"expected xvars and zvars to have the same 2nd dimension. Instead, got {N} and {M}")

    @property
    def slack(self):
        return self.__slack

    @slack.setter
    def slack(self, val):
        if (type(val) is int or type(val) is float) and val >= 0.:
            self.__slack = val
        else:
            raise ValueError("slack must be a nonnegative numeric value")
    
    @property
    def cost_weight(self):
        warnings.warn(f"{type(self).__name__} does not have an associated cost")

    @cost_weight.setter
    def cost_weight(self, val):
        warnings.warn(f"{type(self).__name__} does not have an associated cost. The value is ignored.")

class CostRelaxedNonlinearComplementarity(ComplementarityFunction):
    """
        For the nonlinear complementarity constraint:
            f(x) >= 0
            z >= 0
            z*f(x) = 0
        This class implements the nonnegative inequality constraints as the constraint:
            f(x) >= 0
            z >= 0
        and provides a separate method call to include the product constraint:
            z*f(x) = 0
        as a cost. The parameter cost_weight sets the penalty parameter for the cost function 
    """
    def __init__(self, fcn=None, xdim=0, zdim=1):
        """ initialize the constraint"""
        self.fcn = fcn
        self.xdim = xdim
        self.zdim = zdim
        self.name = self.fcn.__name__
        self.cost_weight = 1.
    
    def eval(self, vars):
        """
        Evaluate the inequality constraints only
        
        The argument must be a numpy array of decision variables ordered as [x, z]
        """
        x, z = np.split(vars, [self.xdim])
        return np.concatenate([self.fcn(x), z], axis=0)

    def lower_bound(self):
        """Return the lower bound"""
        return np.zeros((2*self.zdim))

    def upper_bound(self):
        """Return the upper bound"""
        return np.full((2*self.zdim,), np.inf)

    def product_cost(self, vars):
        """
        Returns the product constraint as a scalar for use as a cost

        The argument must be a numpy array of decision variables organized as [x, z]
        """
        x, z = np.split(vars, [self.xdim])
        return self.cost_weight * z.dot(self.fcn(x))

    def addToProgram(self, prog, xvars, zvars):
        """Add the constraint to the a mathematical program"""


    @property
    def slack(self):
        return None

    @slack.setter
    def slack(self, val):
        warnings.warn(f"{type(self).__name__} does not support setting constant slack variables. The value is ignored.")

    @property
    def cost_weight(self):
        return self.__cost_weight

    @cost_weight.setter
    def cost_weight(self, val):
        if (type(val) == int or type(val) == float) and val >= 0.:
            self.__cost_weight = val
        else:
            raise ValueError("cost_weight must be a nonnegative numeric value")       

class ConstantSlackNonlinearComplementarity(ComplementarityFunction):
    """
    Implements the nonlinear complementarity constraint with a constant slack variable
    Implements the problem as:
        f(x) >= 0
        z >= 0 
        z*f(x) <= s
    In this implementation, the slack is pushed to the upper bound of the constraints
    """
    def eval(self, vars):
        """ 
        Evaluates the original nonlinear complementarity constraint 
        
        The argument must be a numpy array of decision variables organized as [x, z]
        """
        x, z = np.split(vars, [self.xdim])
        fcn_val = self.fcn(x)
        return np.concatenate([fcn_val, z, fcn_val*z], axis=0)

    def upper_bound(self):
        """ Returns the upper bound of the constraint"""
        return np.concatenate([np.full((2*self.zdim, ), np.inf), self.slack * np.ones((self.zdim,))], axis=0)

    def lower_bound(self):
        """ Returns the lower bound of the constraint"""
        return np.concatenate([np.zeros((2*self.zdim,)), -np.full((self.zdim,), np.inf)], axis=0)

class VariableSlackNonlinearComplementarity(ComplementarityFunction):
    """
    Implements the nonlinear complementarity constraint as
        f(x) >= 0
        z >= 0 
        z*f(x) - s <= 0
    where s is a decision variable. In this implementation, the bounds on the constraint are fixed
    """
    def eval(self, vars):
        """
        Evaluate the complementarity constraint
        
        The argument must be a numpy array of decision variables [x, z, s]
        """
        x, z, s = np.split(vars, np.cumsum([self.xdim, self.zdim]))
        fcn_val = self.fcn(x)
        return np.concatenate([fcn_val, z, fcn_val*z -s], axis=0)

    def upper_bound(self):
        """Return the lower bound"""
        return np.concatenate([np.full((2*self.zdim,), np.inf), np.zeros((self.zdim,))], axis=0)

    def lower_bound(self):
        """Return the upper bound"""
        return np.concatenate([np.zeros((2*self.zdim,)), -np.full((self.zdim,), np.inf)], axis=0)

class ConstantSlackLinearEqualityComplementarity(ComplementarityFunction):
    """
    Introduces new variables and an equality constraint to implement the nonlinear constraint as a linear complementarity constraint with a nonlinear equality constraint. The original problem is implemented as:
        r - f(x) = 0
        r >= 0, z>= 0
        r*z <= s
    where r is the extra set of variables and s is a constant slack added to the upper bound
    """
    def eval(self, vars):
        """
        Evaluate the constraint
        
        The argument must be a numpy array of decision variables ordered as [x, z, r]
        """
        x, z, r = np.split(vars, np.cumsum([self.xdim, self.zdim]))
        fcn_val = self.fcn(x)
        return np.concatenate((r-fcn_val, r, z, r*z), axis=0)

    def upper_bound(self):
        """Return the upper bound of the constraint"""
        return np.concatenate([np.zeros((self.zdim,)), np.full((2*self.zdim,), np.inf), self.slack*np.ones((self.zdim,))], axis=0)

    def lower_bound(self):
        """Return the lower bound of the constraint"""
        return np.concatenate([np.zeros((3*self.zdim,)), -np.full((self.zdim,), np.inf)], axis=0)

class VariableSlackLinearEqualityComplementarity(ComplementarityFunction):
    """
    Introduces new variables and an equality constraint to implement the nonlinear constraint as a linear complementarity constraint with a nonlinear equality constraint. The original problem is implemented as:
        r - f(x) = 0
        r >= 0, z>= 0
        r*z -s <= 0
    where r is the extra set of variables and s is a variable slack
    """
    def eval(self, vars):
        """
        Evaluate the constraint. 
        The arguments must be a numpy array of decision variables including:
            [x, z, r, s]
        """
        x, z, r, s = np.split(vars, np.cumsum([self.xdim, self.zdim]))
        fcn_val = self.fcn(x)
        return np.concatenate((r-fcn_val, r, z, r*z - s), axis=0)

    def upper_bound(self):
        """Return the upper bound of the constraint"""
        return np.concatenate([np.zeros((self.zdim,)), np.full((2*self.zdim,), np.inf), np.zeros((self.zdim,))], axis=0)

    def lower_bound(self):
        """Return the lower bound of the constraint"""
        return np.concatenate([np.zeros((3*self.zdim,)), -np.full((self.zdim,), np.inf)], axis=0)

class NonlinearComplementarityFcn():
    """
    Implements a complementarity relationship involving a nonlinear function, such that:
        f(x) >= 0
        z >= 0
        f(x)*z <= s
    where f is the function, x and z are decision variables, and s is a slack parameter.
    By default s = 0 (strict complementarity)
    """
    def __init__(self, fcn, xdim=0, zdim=1):
        self.fcn = fcn
        self.xdim = xdim
        self.zdim = zdim
        self.slack = 0.
    
    def __call__(self, vars):
        """Evaluate the complementarity constraint """
        x, z, s = self.split_vars(vars, [self.xdim, self.zdim])
        fcn_val = self.fcn(x)
        return np.concatenate((fcn_val,z, fcn_val * z - self.slack), axis=0)

    def lower_bound(self):
        return np.concatenate((np.zeros((2*self.zdim,)), -np.full((self.zdim,), np.inf)), axis=0)
    
    def upper_bound(self):
        return np.concatenate((np.full((2*self.zdim,), np.inf), np.zeros((self.zdim,))), axis=0)

    @property
    def slack(self):
        return self.__slack

    @slack.setter
    def slack(self, val):
        if (type(val) == int or type(val) == float) and val >= 0.:
            self.__slack = val
        else:
            raise ValueError("slack must be a nonnegative numeric value")