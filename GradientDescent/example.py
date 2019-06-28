from GradientDescent.NesterovDescent import NesterovAcceleratedGradient
from GradientDescent.CoordinateDescent import CoordinateGradientDescent
from GradientDescent.SteepestDescent import SteepestGradientDescent

""" Three-Hump Camel Function (thcf) is taken in order to find global minimum.
    The function depends on two variables and features a valley.
"""


def thcf(x):
    """Standard equation of the Three-Hump Camel Function.
    """
    eqn = 2 * (x[0] ** 2) - 1.05 * (x[0] ** 4) + \
        (x[0] ** 6) / 6 + x[0] * x[1] + (x[1] ** 2)
    return eqn                                      # Function to minimise


def thcf_partial1(x):
    """ Partial derivative of the Three-Hump Camel Function with respect to x1.
    """
    partial = x[1] + 4 * x[0] - 4.2 * (x[0] ** 3) + (x[0] ** 5)
    return partial                                  # Gradient 1


def thcf_partial2(x):
    """Partial derivative of the Three-Hump Camel Function with respect to x2.
    """
    partial = x[0] + 2 * x[1]
    return partial                                  # Gradient 2


one = NesterovAcceleratedGradient([5, 5], [5, 5], [5, 5],
                                  1e-3, 1e-6, 1, 10000, 0.4)
two = one.graph_full_nesterov(thcf, thcf_partial1, thcf_partial2)
