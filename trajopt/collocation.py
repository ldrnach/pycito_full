"""
collocation.py: Tools for polynomial collocation methods, focusing mainly on lagrange interpolating polynomials and orthogonal collocation methods

Luke Drnach
September 28, 2021
"""

import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt
#TODO: Unittest RadauCollocation.right_endpoint_weights
class LagrangeBasis():
    """
    Implementation of the Lagrange Basis Polynomial
    
    :fieldname: nodes - list or array of knot points at which the polynomial evaluates to zero (or one, for node[centerindex])
    :fieldname: centerindex - integer index of the node at which the polynomial evaluates to one
    :fieldname: weight - scalar normalizing weight for the polynomial
    
    See also: LagrangeInterpolant
    """
    def __init__(self, nodes, centerindex=0):
        """
        Construct a single Lagrange Basis Polynomial. The polynomial evaluates to zero at all nodes except the center node, where it evaluates to 1

        :param nodes: list or numpy array of node points 
        :param centerindex: int, index for the node at which the basis polynomial evaluates to 1 (default: 0)
       
        :returns: The constructed Lagrange basis polynomial 
        """
        self._nodes = np.asarray(nodes)
        self.centerindex = centerindex
        self.weight = self._calculate_weight(nodes, centerindex)

    @staticmethod
    def _calculate_weight(nodes, centerindex=0):
        """
        Internal method for calculating the normalizing weight from the nodes
        """
        diffs = np.asarray([nodes[centerindex] - node for node in nodes if node != nodes[centerindex]])       
        return 1/np.prod(diffs)

    def eval(self, x_list):
        """
        Evaluate the Lagrange basis polynomial at a list of specific values
        
        :param x_list: a N-list or (N,)-array of values at which to evaluate the polynomial

        :returns: (N,) numpy array of values of the basis polynomial
        """

        x_list = np.atleast_1d(x_list)
        return np.asarray([self._eval(x) for x in x_list])

    def _eval(self, x):
        """
        Evaluate the Lagrange basis polynomial at the specific value
        
        _eval assumes only one point is given, that x has only one element
        """
        diffs = [x - node for node in self.nodes if node != self.centernode]
        return self.weight*np.prod(diffs) 

    def derivative(self, x_list):
        """
        Evaluate the derivative of the Lagrange basis polynomial at a list of specific values
        
        :param x_list: a N-list or (N,)-array of values at which to evaluate the polynomial

        :returns: (N,) numpy array of derivatives of the basis polynomial evaluated at x_list
        """
        x_list = np.atleast_1d(x_list)
        return np.asarray([self._derivative(x) for x in x_list])

    def _derivative(self, x):
        """
        Evaluates the derivative of the Lagrange basis polynomial at a single value

        _derivative assumes only one point is given / that x has only one element
        """
        diffs = np.asarray([x - node for node in self.nodes if node != self.centernode])
        # sumprod = 0
        # for n in range(diffs.shape[0]):
        #     d = diffs.copy()
        #     d[n] = 1
        #     sumprod += np.prod(d)
        # return self.weight * sumprod
        leftprods = np.ones_like(diffs)
        rightprods = np.ones_like(diffs)
        for n in range(1, diffs.shape[0]):
            leftprods[n] = leftprods[n-1] * diffs[-n]
            rightprods[-n-1] = rightprods[-n] * diffs[n-1]
        return self.weight * leftprods.dot(rightprods)

    @property
    def centernode(self):
        """
        :returns: the node for which the polynomial evaluates to one
        """
        return self.nodes[self.centerindex]

    @property
    def nodes(self):
        """
        :returns: a (N,) numpy array of node values
        """
        return self._nodes

    @nodes.setter
    def nodes(self, val):
        """
        Set the values of the nodes.
        The new nodes must be an array of the same shape as the previous nodes
        """
        val = np.asarray(val)
        if val.shape != self._nodes.shape:
            raise ValueError(f"nodes must be a numpy array with shape {self._nodes.shape}")
        self._nodes = val
        self.weight = self._calculate_weight(self._nodes, self._centerindex)

class LagrangeInterpolant():
    """
    Implementation of the Lagrange Interpolating Polynomial
    
    :fieldname: nodes - (N,)-list or array of knot points at which the function values are given
    :fieldname: values - list or array of function values to interpolate between. Array has shape (N,) for scalar interpolation, and (M,N) for M-vector interpolation
    :fieldname: bases - (N,)-list of Lagrange basis polynomials comprising the interpolating polynomial
    :fieldname: differentiation_matrix - (N, N)-array converting the given function values to their derivatives
    
    See also: LagrangeInterpolant
    """
    def __init__(self, nodes, values):
        """
        Construct a Lagrange interpolating polynomial, the polynomial of least degree that interpolates the dataset

        :param nodes: (N,)-list or numpy array of node points 
        :param values: (N,)-list or numpy array (scalar interpolation) or (M,N)-list or array (vector interpolation) of function values
       
        :returns: The constructed Lagrange interpolating polynomial
        """
        self._bases = [LagrangeBasis(nodes, i) for i in range(len(nodes))]
        val = np.asarray(values)
        self._values = val
        self._differentiation_matrix = self._calculate_differentiation_matrix()

    def eval(self, x_list):
        """
        Evaluate the Lagrange interpolant at a list of specific values
        
        :param x_list: a N-list or (N,)-array of values at which to evaluate the polynomial

        :returns: (N,) numpy array of values of the interpolating polynomial, or an (M,N) array for vector interpolation
        """
        x_list = np.atleast_1d(x_list)
        return np.stack([self._eval(x) for x in x_list]).transpose()

    def _eval(self, x):
        """
        Evaluate the the Lagrange interpolant at a single point
        
        :param x: scalar point at which to evaluate the interpolant

        :returns: (M,)-numpy array of interpolant values, where M is the output dimension of the function
        """
        x = np.asarray(x)
        # Return the datapoint
        if x in self.nodes:
            return self.values[..., np.where(self.nodes==x)[0].item()]
        # Interpolate
        weights = np.asarray([basis.weight/(x - basis.centernode) for basis in self.bases])
        total = np.sum(weights)
        scaled_total = self.values.dot(weights)
        return np.asarray(scaled_total / total)

    def derivative(self, x_list):
        """
        Evaluate the derivative of the Lagrange interpolant at a list of specific values
        
        :param x_list: a N-list or (N,)-array of values at which to evaluate the derivative of the polynomial

        :returns: (N,) numpy array of values of the derivative, or an (M,N) array for vector interpolation
        """
        x_list = np.atleast_1d(x_list)
        return np.stack([self._derivative(x) for x in x_list]).transpose()

    def _derivative(self, x):
        """
        Evaluate the derivative of the Lagrange interpolant at a single point
        
        :param x: scalar point at which to evaluate the derivative

        :returns: (M,)-numpy array of derivative values, where M is the output dimension of the function
        """
        derivs = np.concatenate([basis.derivative(x) for basis in self.bases])
        return self.values.dot(derivs)

    def _calculate_differentiation_matrix(self):
        """Calculate the matrix for calculating derivatives at the evaluation points"""
        derivs = [basis.derivative(self.nodes) for basis in self.bases]
        return np.asarray(derivs).transpose()

    @property
    def nodes(self):
        """Nodes at which the interpolant gives exact values"""
        return self.bases[0].nodes

    @nodes.setter
    def nodes(self, val):
        for n in range(len(self.bases)):
            self.bases[n].nodes = val
        self.differentiation_matrix = self._calculate_differentiation_matrix()

    @property
    def values(self):
        """Exact values of the underlying function"""
        return self._values

    @values.setter
    def values(self, val):
        val = np.asarray(val)
        if val.shape[-1] != self._values.shape[-1]:
            raise ValueError(f"values must contain {self._values.shape[-1]} elements")
        self._values = val

    @property
    def bases(self):
        """List of Lagrange basis polynomials"""
        return self._bases

    @property
    def differentiation_matrix(self):
        """Returns the differentiation matrix for converting values at node points to their derivatives"""
        return self._differentiation_matrix

class RadauCollocation(LagrangeInterpolant):
    """
    Constructs a Lagrange Interpolating polynomial using the roots of the Gauss-Jacobi Orthogonal polynomials as the knot points
    
    For a Kth order interpolation, Radau Collocation uses the roots of the (K-1)th order Gauss-Jacobi polynomials on the interval [0, 1], as well as the end point 1 as the node points
    
    Radau collocation does not require sample values to construct the object, and assumes the underlying function is scalar valued at takes its roots at the node points. For interpolating known functions, values can be provided after object construction, by evaluting the function at the node points of the collocation polynomial.
    
    """
    def __init__(self, order=3, domain=[0,1], values=None):
        """
        Construct a Lagrange Interpolant using Radau collocation points
        
        :param order:  (optional)  int, specifies the order of the polynomial interpolation (default: 3)
        :param domain: (optional)  2-list or array of finite floats, defines the interval on which the interpolation is valid (default: [0,1])
        :param values: (optional)  array of values of the underlying function at the node points (default: zeros)

        :returns: a RadauCollocation object
        """
        nodes, _ = special.roots_sh_jacobi(order, 2, 1)
        nodes = np.concatenate([nodes, np.ones((1,))], axis=0)
        self.initial_time = domain[0]
        self.timestep = domain[1] - domain[0]
        if values == None:
            values = np.zeros((order+1, ))    
        super(RadauCollocation, self).__init__(nodes, values)
       
    def eval(self, node):
        """
        Evaluate the Radau Collocation interpolant at a list of specific values. 
        
        Nodes are converted to normalized time (i.e. the interval [0,1]) before the interpolant is evaluated.

        :param node: a N-list or (N,)-array of values at which to evaluate the polynomial, given in non-normalized time

        :returns: (N,) numpy array of values of the interpolating polynomial, or an (M,N) array for vector interpolation
        """
        return super().eval(self.map_to_normalized_time(node))

    def derivative(self, node):
        """
        Evaluate the derivative of the Radau collocation interpolant at a list of specific values. 
        
        Nodes are converted to normalized time (i.e. the interval [0,1]) before the derivative is evaluated.
                
        :param node: a N-list or (N,)-array of values at which to evaluate the derivative, given in non-normalized time

        :returns: (N,) numpy array of values of the interpolating polynomial, or an (M,N) array for vector interpolation
        """
        return super().derivative(self.map_to_normalized_time(node))

    def map_to_normalized_time(self, node):
        """
        Map the node value to normalized time

        The given value is assumed to be in the original problem domain, and is mapped to the interval [0,1]
        
        :param node: an N-list (N,)-array of node values in the original problem domain

        :returns: an N-list or (N,)-array of node values mapped to the interval [0,1]

        See also: map_to_full_time
        """
        return (node - self.initial_time)/self.timestep

    def map_to_full_time(self, node):
        """
        Maps normalized node values to the original problem domain

        The given value is assumed to be in the interval [0,1], and is mapped to the original problem domain
        
        :param node: an N-list (N,)-array of node values in the interval [0,1]

        :returns: an N-list or (N,)-array of node values mapped to the original problem domain

        See also: map_to_normalized_time
        """
        return self.initial_time + node * self.timestep

    def _calculate_differentiation_matrix(self):
        """Get the matrix for calculating derivatives at the evaluation points"""
        derivs = [basis.derivative(self.nodes) for basis in self.bases]
        return np.asarray(derivs).transpose()

    def right_endpoint_weights(self):
        weights = [basis.eval(1) for basis in self.bases]
        return np.asarray(weights)

    def left_endpoint_weights(self):
        return np.asarray([basis.eval(0) for basis in self.bases])

    @property
    def nodes(self):
        nodes_ = super(RadauCollocation, self).nodes
        return self.map_to_full_time(nodes_)

    @nodes.setter
    def nodes(self, val):
        super(RadauCollocation, self).nodes = self.map_to_normalized_time(val)

    @property
    def order(self):
        return self.nodes.shape[0] - 1

def witch_of_agnesi(x):
    return 1./(1 + 25 * x**2)

def runge_example():
    # Generate Runge's function, the Witch of Agensi
    domain = np.linspace(-1, 1, 1000)
    runge = witch_of_agnesi(domain)
    # 9th order interpolant (8th order poly), equally spaced nodes
    eq_nodes = np.linspace(-1, 1, 9)
    eq_values = witch_of_agnesi(eq_nodes)
    eq_poly = LagrangeInterpolant(eq_nodes, eq_values)
    eq_interp = eq_poly.eval(domain)
    # 9th order interpolant (8th order poly), Radau collocation nodes
    radau_poly = RadauCollocation(order=9, domain=[-1,1])
    radau_nodes = radau_poly.nodes
    radau_poly.values = witch_of_agnesi(radau_nodes)
    radau_interp = radau_poly.eval(domain)
    # Plot all the results
    _, axs = plt.subplots(1,1)
    axs.plot(domain, runge, linewidth=1.5, label="Runge's function")
    axs.plot(domain, eq_interp,'g-', linewidth=1.5, label="Equidistant Nodes")
    axs.plot(eq_nodes, eq_values,'go')
    axs.plot(domain, radau_interp, 'r-', linewidth=1.5, label="Radau Nodes")
    axs.plot(radau_nodes, radau_poly.values,'ro')
    axs.legend()
    plt.show()

if __name__ == '__main__':
    runge_example()