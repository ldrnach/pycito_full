"""
collocation.py: Tools for polynomial collocation methods, focusing mainly on lagrange interpolating polynomials and orthogonal collocation methods

Luke Drnach
September 28, 2021
"""

import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt

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
        leftprods = np.ones_like(diffs)
        rightprods = np.ones_like(diffs)
        for n in range(1, diffs.shape[0]):
            leftprods[n] = leftprods[n-1] * diffs[n]
            rightprods[-1-n] = rightprods[-n] * diffs[-1-n]
        return self.weight * leftprods.dot(rightprods)

    @property
    def centernode(self):
        """
        :returns: the node for which the polynomial evaluates to one
        """
        return self.nodes[self.centerindex]

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, val):
        val = np.asarray(val)
        if val.shape != self._nodes.shape:
            raise ValueError(f"nodes must be a numpy array with shape {self._nodes.shape}")
        self._nodes = val

class LagrangeInterpolant():
    """
    Implementation of the Lagrange Interpolating Polynomial
    
    :fieldname: bases - list or array of knot points at which the polynomial evaluates to zero (or one, for node[centerindex])
    :fieldname: values - integer index of the node at which the polynomial evaluates to one
    :fieldname: weight - scalar normalizing weight for the polynomial
    
    See also: LagrangeInterpolant
    """
    def __init__(self, nodes, values):
        self._bases = [LagrangeBasis(nodes, i) for i in range(len(nodes))]
        val = np.asarray(values)
        # if val.ndim == 1:
        #     val = np.expand_dims(val, axis=0)
        self._values = val
        self._nodes = np.asarray(nodes)
        self._differentiation_matrix = self._calculate_differentiation_matrix()

    def eval(self, x_list):
        """
        Evaluate the Lagrange Interpolant at the specific value
        
        Arguments:
            x_list: a list or array of values at which to evaluate the polynomial
        """
        x_list = np.atleast_1d(x_list)
        return np.stack([self._eval(x) for x in x_list]).transpose()

    def _eval(self, x):
        """
        Evaluate the Lagrange Interpolant at the specific value
        
        _eval assumes only one point is given, that x has only one element
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
        Evaluate the derivative of the Lagrange Interpolant at the specific values
        
        x_list: a list or array of points at which to evaluate the derivative
        """
        x_list = np.atleast_1d(x_list)
        return np.stack([self._derivative(x) for x in x_list]).transpose()

    def _derivative(self, x):
        """
        Evaluate the deriative of the Lagrange Interpolant at a single point

        x: a single element at which to evaluate the derivative
        """
        derivs = np.concatenate([basis.derivative(x) for basis in self.bases])
        return self.values.dot(derivs)

    def _calculate_differentiation_matrix(self):
        """Get the matrix for calculating derivatives at the evaluation points"""
        derivs = [basis.derivative(self.nodes) for basis in self.bases]
        return np.asarray(derivs).transpose()

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, val):
        val = np.asarray(val)
        if val.shape != self._nodes.shape:
            raise ValueError(f"nodes must be a numpy array with shape {self._nodes.shape}")
        self._nodes = val

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, val):
        val = np.asarray(val)
        if val.shape != self._values.shape:
            raise ValueError(f"values must be a numpy array with shape {self._values.shape}")
        self._values = val

    @property
    def bases(self):
        return self._bases

    @property
    def differentiation_matrix(self):
        return self._differentiation_matrix


class RadauCollocation(LagrangeInterpolant):
    def __init__(self, order=3, domain=[0,1]):
        """Use Lagrange Interpolation on the interval [0,1] using the roots of the Gauss-Jacobi Orthogonal polynomials"""
        nodes, _ = special.roots_sh_jacobi(order-1, 2, 1)
        nodes = np.concatenate([nodes, np.ones((1,))], axis=0)
        self.initial_time = domain[0]
        self.timestep = domain[1] - domain[0]
        super(RadauCollocation, self).__init__(nodes, values=np.zeros((order,)))
       
    def eval(self, node):
        """Evaluate the polynomial at the specified node"""
        return super().eval(self.map_to_normalized_time(node))

    def derivative(self, node):
        """Evaluate the derivative of the polynomial at the specified node"""
        return super().derivative(self.map_to_normalized_time(node))

    def map_to_normalized_time(self, node):
        """
        Map the node to normalized time
        
        If the node is in the original domain, the normalized time node is within the interval [0,1]
        """
        return (node - self.initial_time)/self.timestep

    def map_to_full_time(self, node):
        return self.initial_time + node * self.timestep

    def _calculate_differentiation_matrix(self):
        """Get the matrix for calculating derivatives at the evaluation points"""
        derivs = [basis.derivative(self.nodes) for basis in self.bases]
        return np.asarray(derivs).transpose()

    @property
    def nodes(self):
        return self.map_to_full_time(self._nodes)

    @nodes.setter
    def nodes(self, val):
        self._nodes = self.map_to_normalized_time(val)

    @property
    def order(self):
        return self.nodes.shape[0]


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