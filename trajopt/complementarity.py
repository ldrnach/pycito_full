"""
complementarity: implementations of complementarity constraints for use in optimization problems.

Luke Drnach
June 11, 2021
"""
import numpy as np
import warnings
from abc import ABC, abstractmethod

class ComplementarityConstraint(ABC):
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
        self.__slack = 0.
        self.name = self.fcn.__name__

    def __call__(self, vars):
        """Evaluate the constraint"""
        return self.eval(vars)

    def str(self):
        return f"{type(self).__name__} on the function {self.fcn.__name__} with input dimension {self.xdim} and output dimension {self.zdim}\n"

    def set_description(self, name=None):
        """Set a string description for the constraint"""
        if name is None:
            self.name = self.fcn.__name__
        else:
            self.name = name

    @abstractmethod
    def eval(self, vars):
        """Concrete implementation of evaluator"""

    @abstractmethod
    def lower_bound(self):
        """Returns the lower bound of the constraint"""

    @abstractmethod
    def upper_bound(self):
        """Returns the upper bound of the constraint"""

    def addToProgram(self, prog, xvars, zvars):
        """ Add the complementarity constraint to a mathematical program """
        xvars, zvars = self._check_vars(xvars, zvars)
        dvars = np.concatenate([xvars, zvars], axis=0)
        for n in range(dvars.shape[1]):
            prog.AddConstraint(self, lb=self.lower_bound(), ub=self.upper_bound(), vars=dvars[:,n], description=self.name)

    def _check_vars(self, xvars, zvars):
        """ Check that the decision variables are appropriately sized"""
        if np.ndim(xvars) == 1:
            xvars = np.expand_dims(xvars, axis=1)
        if np.ndim(zvars) == 1:
            zvars = np.expand_dims(zvars, axis=1)
        nX, N = xvars.shape
        nZ, M = zvars.shape
        if nX != self.xdim:
            raise ValueError(f"Expected {self.xdim} xvars but got {nX} instead")
        if nZ != self.zdim:
            raise ValueError(f"Expected {self.zdim} zvars but got {nZ} instead")
        if N != M:
            raise ValueError(f"expected xvars and zvars to have the same 2nd dimension. Instead, got {N} and {M}")
        return xvars, zvars

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

class NonlinearConstantSlackComplementarity(ComplementarityConstraint):
    """
    Implements the nonlinear complementarity constraint with a constant slack variable
    Implements the problem as:
        f(x) >= 0
        z >= 0 
        z*f(x) <= s
    In this implementation, the slack is pushed to the upper bound of the constraints
    """
    def __init__(self, fcn, xdim=1, zdim=1, slack=0.):
        self._prog = None
        super(NonlinearConstantSlackComplementarity, self).__init__(fcn, xdim, zdim)
        self.slack = slack

    def str(self):
        text = super(NonlinearConstantSlackComplementarity, self).str()
        text += f"\tConstant Slack = {self.slack}\n"
        return text

    def eval(self, dvars):
        """ 
        Evaluates the original nonlinear complementarity constraint 
        
        The argument must be a numpy array of decision variables organized as [x, z]
        """
        x, z = np.split(dvars, [self.xdim])
        fcn_val = self.fcn(x)
        return np.concatenate([fcn_val, z, fcn_val*z], axis=0)

    def lower_bound(self):
        """ Returns the lower bound of the constraint"""
        return np.concatenate([np.zeros((2*self.zdim,)), np.full((self.zdim,), -np.inf)], axis=0)

    def upper_bound(self):
        """ Returns the upper bound of the constraint"""
        return np.concatenate([np.full((2*self.zdim, ), np.inf), self.slack * np.ones((self.zdim,))], axis=0)

    def addToProgram(self, prog, xvars, zvars):
        """Add the complementarity constraint to a mathematical program"""
        super(NonlinearConstantSlackComplementarity, self).addToProgram(prog, xvars, zvars)
        self._prog = prog

    @property
    def slack(self):
        return self.__slack

    @slack.setter
    def slack(self, val):
        if (type(val) is int or type(val) is float) and val >= 0.:
            self.__slack = val
            if self._prog is not None:
                ub = np.concatenate([np.full((2*self.zdim, ), np.inf), val * np.ones((self.zdim,))], axis=0)
                for cstr in self._prog.GetAllConstraints():
                    if cstr.evaluator().get_description() == self.name:
                        cstr.evaluator().UpdateUpperBound(new_ub = ub)
        else:
            raise ValueError("slack must be a nonnegative numeric value")

class NonlinearVariableSlackComplementarity(ComplementarityConstraint):
    """
    Implements the nonlinear complementarity constraint as
        f(x) >= 0
        z >= 0 
        z*f(x) - s <= 0
    where s is a decision variable. In this implementation, the bounds on the constraint are fixed, but 
    the slack variable s is introduced and minimized in an associated cost.
    """
    def __init__(self, fcn, xdim=1, zdim=1):
        super(NonlinearVariableSlackComplementarity, self).__init__(fcn, xdim, zdim)
        self.__slack_cost = []
        self.__cost_weight = 1.
        self.__slack = None

    def str(self):
        text = super(NonlinearVariableSlackComplementarity, self).str()
        text += f"\tVariable slack cost weight = {self.__cost_weight}\n"
        return text

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

    def addToProgram(self, prog, xvars, zvars):
        """ Add the constraint with the slack variables to the mathematical program"""
        xvars, zvars = self._check_vars(xvars, zvars)
        new_slack = prog.NewContinuousVariables(rows=1, cols=zvars.shape[1], name=self.name+"_slacks")
        dvars = np.concatenate([xvars, zvars, new_slack], axis=0)
        for n in range(dvars.shape[1]):
            prog.AddConstraint(self, 
                            lb = self.lower_bound(),
                            ub = self.upper_bound(),
                            vars = dvars[:,n],
                            description = self.name)
        # Add a cost on the slack variables
        new_cost = prog.AddLinearCost(a = self.cost_weight * np.ones((new_slack.shape[1]),), b=np.zeros((1,)), vars = new_slack)
        new_cost.evaluator().set_description(self.name + "SlackCost")
        # concatenate the associated variables and cost
        self.__slack_cost.append(new_cost)
        if self.__slack is None:
            self.__slack = new_slack
        else:
            self.__slack = np.row_stack([self.__slack, new_slack])

    @property
    def slack(self):
        return self.__slack

    @slack.setter
    def slack(self, val):
        warnings.warn(f"{type(self).__name__} does not allow setting constant slack variables. The value is ignored")

    @property
    def cost_weight(self):
        return self.__cost_weight

    @cost_weight.setter
    def cost_weight(self, val):
        if (type(val) == int or type(val) == float) and val >= 0.:
            #Store the cost weight value
            self.__cost_weight = val
            # Update slack cost
            if self.__slack_cost is not None:
                for cost in self.__slack_cost:
                    nvars = cost.variables().size
                    cost.evaluator().UpdateCoefficients(new_a = val*np.ones((nvars,)))
        else:
            raise ValueError("cost_weight must be a nonnegative numeric value")

class CostRelaxedNonlinearComplementarity(ComplementarityConstraint):
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
        """
            Creates the complementarity constraint function. Stores a reference to the nonlinear function, the dimensions of the x and z variables, and sets the slack to 0
        """
        self.fcn = fcn
        self.xdim = xdim
        self.zdim = zdim
        self.dvars = None
        self.name=self.fcn.__name__
        self.cost_weight = 1.
    
    def str(self):
        text = super(CostRelaxedNonlinearComplementarity, self).str()
        text += f"\tCost weight: {self.cost_weight}"
        return text

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
        xvars, zvars = self._check_vars(xvars, zvars)
        dvars = np.concatenate([xvars, zvars], axis=0)
        for n in range(dvars.shape[1]):
            prog.AddConstraint(self, lb = self.lower_bound(), ub=self.upper_bound(), vars=dvars[:,n], description=self.name)
            prog.AddCost(self.product_cost, vars=dvars[:,n], description=self.name+"Cost")

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

class LinearEqualityConstantSlackComplementarity(ComplementarityConstraint):
    """
    Introduces new variables and an equality constraint to implement the nonlinear constraint as a linear complementarity constraint with a nonlinear equality constraint. The original problem is implemented as:
        r - f(x) = 0
        r >= 0, z>= 0
        r*z <= s
    where r is the extra set of variables and s is a constant slack added to the upper bound
    """
    def __init__(self, fcn, xdim=1, zdim=1, slack=0):
        self.fcn = fcn
        self.xdim = xdim
        self.zdim = zdim
        self.__const_slack = slack     
        self._prog = None
        self.name = fcn.__name__
        self.__slack_vars = None

    def str(self):
        text = super(LinearEqualityConstantSlackComplementarity, self).str()
        text += f"\tConstant Slack: {self.__const_slack}\n"
        return text

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
        return np.concatenate([np.zeros((self.zdim,)), np.full((2*self.zdim,), np.inf), self.__const_slack*np.ones((self.zdim,))], axis=0)

    def lower_bound(self):
        """Return the lower bound of the constraint"""
        return np.concatenate([np.zeros((3*self.zdim,)), -np.full((self.zdim,), np.inf)], axis=0)

    def addToProgram(self, prog, xvars, zvars):
        xvars, zvars = self._check_vars(xvars, zvars)
        # Create new slack variables
        new_slacks = prog.NewContinuousVariables(rows = zvars.shape[0], cols=zvars.shape[1], name=self.name + "_slacks")
        # All decision variables
        dvars = np.concatenate([xvars, zvars, new_slacks], axis=0)
        # Add complementarity constraints
        for n in range(dvars.shape[1]):
            prog.AddConstraint(self, lb=self.lower_bound(), ub=self.upper_bound(), vars=dvars[:,n], description=self.name)
        # Store the reference to the program
        self._prog = prog
        # Store the slack variables
        if self.__slack_vars is None:
            self.__slack_vars = new_slacks
        else:
            self.__slack_vars = np.row_stack([self.__slack_vars, new_slacks])

    @property
    def slack(self):
        return self.__slack_vars

    @slack.setter
    def slack(self, val):
        #Set the CONSTANT Slack value
        if type(val) is int or type(val) is float and val >=0:
            self.__const_slack = val
            if self._prog is not None:
                ub = np.concatenate([np.zeros((self.zdim,)), np.full((2*self.zdim,), np.inf), val*np.ones((self.zdim,))], axis=0)
                for cstr in self._prog.GetAllConstraints():
                    if cstr.evaluator().get_description() == self.name:
                        cstr.evaluator().UpdateUpperBound(new_ub = ub)
        else:
            raise ValueError("slack must be a nonnegative numeric value")

class LinearEqualityVariableSlackComplementarity(ComplementarityConstraint):
    """
    Introduces new variables and an equality constraint to implement the nonlinear constraint as a linear complementarity constraint with a nonlinear equality constraint. The original problem is implemented as:
        r - f(x) = 0
        r >= 0, z>= 0
        r*z -s <= 0
    where r is the extra set of variables and s is a variable slack
    """
    def __init__(self, fcn, xdim=1, zdim=1):
       self.fcn = fcn
       self.xdim = xdim
       self.zdim = zdim
       self.name = fcn.__name__
       self.__slack_vars = None
       self.__slack_cost = []
       self.__cost_weight = 1.

    def str(self):
        text = super(LinearEqualityVariableSlackComplementarity, self).str()
        text += f"\tSlack cost weight: {self.__cost_weight}\n"
        return text

    def eval(self, vars):
        """
        Evaluate the constraint. 
        The arguments must be a numpy array of decision variables including:
            [x, z, r, s]
        """
        x, z, r, s = np.split(vars, np.cumsum([self.xdim, self.zdim, self.zdim]))
        fcn_val = self.fcn(x)
        return np.concatenate((r-fcn_val, r, z, r*z - s), axis=0)

    def upper_bound(self):
        """Return the upper bound of the constraint"""
        return np.concatenate([np.zeros((self.zdim,)), np.full((2*self.zdim,), np.inf), np.zeros((self.zdim,))], axis=0)

    def lower_bound(self):
        """Return the lower bound of the constraint"""
        return np.concatenate([np.zeros((3*self.zdim,)), -np.full((self.zdim,), np.inf)], axis=0)

    def addToProgram(self, prog, xvars, zvars):
        xvars, zvars = self._check_vars(xvars, zvars)
        # Create new slack variables
        new_slacks = prog.NewContinuousVariables(rows = zvars.shape[0]+1, cols=zvars.shape[1], name=self.name + "_slacks")
        # Add bounding box constraints
        dvars = np.concatenate([xvars, zvars, new_slacks], axis=0)
        # Add complementarity constraints
        for n in range(dvars.shape[1]):
            prog.AddConstraint(self,lb=self.lower_bound(), ub=self.upper_bound(), vars=dvars[:,n], description=self.name)
        # Add a cost on the slack variables
        new_cost = prog.AddLinearCost(a = self.cost_weight * np.ones((new_slacks.shape[1]),), b=np.zeros((1,)), vars = new_slacks[-1,:])
        new_cost.evaluator().set_description(self.name+"SlackCost")
        self.__slack_cost.append(new_cost)
        # Store the variables
        if self.__slack_vars is not None:
            self.__slack_vars = np.row_stack([self.__slack_vars, new_slacks])
        else:
            self.__slack_vars = new_slacks
    
    @property
    def slack(self):
        return self.__slack_vars

    @slack.setter
    def slack(self, val):
        warnings.warn(f"{type(self).__name__} does not allow setting constant slack variables. The value is ignored")

    @property
    def cost_weight(self):
        return self.__cost_weight

    @cost_weight.setter
    def cost_weight(self, val):
        if (type(val) == int or type(val) == float) and val >= 0.:
            #Store the cost weight value
            self.__cost_weight = val
            # Update slack cost
            for cost in self.__slack_cost:
                nvars = cost.variables().size
                cost.evaluator().UpdateCoefficients(new_a = val*np.ones(nvars))
        else:
            raise ValueError("cost_weight must be a nonnegative numeric value")

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

    def str(self):
        return f"{type(self).__name__} on function {self.fcn.__name__} with input dimension {self.xdim} and output dimension {self.zdim}\n"

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