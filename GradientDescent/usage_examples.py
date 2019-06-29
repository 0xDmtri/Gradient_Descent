from GradientDescent.NesterovDescent import NesterovAcceleratedGradient
from GradientDescent.CoordinateDescent import CoordinateGradientDescent
from GradientDescent.SteepestDescent import SteepestGradientDescent

""" Three-Hump Camel Function (thcf) is taken in order to find global minimum
    and demostrate the capability of the algorithm.

    This particular function depends on two variables and features a valley.
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


# Nesterov Accelerated Gradient Usage Examples:

nag = NesterovAcceleratedGradient([5, 5], [5, 5], [5, 5],
                                  1e-3, 1e-6, 1, 10000, 0.4)
graph_full_function_via_nag_class = nag.graph_full_function(thcf)
graph_interval_function_via_nag_class = nag.graph_interval_function(thcf)
calc_nag = nag.calculate(thcf, thcf_partial1, thcf_partial2)
graph_full_nag = nag.graph_full_nesterov(thcf, thcf_partial1, thcf_partial2)
graph_partial_nag = nag.graph_partial_nesterov(thcf, thcf_partial1,
                                               thcf_partial2)

# Coordinate Gradient Descent Usage Examples:

cgd = CoordinateGradientDescent([5, 5], [5, 5], [5, 5],
                                1e-3, 1e-6, 1, 10000)
graph_full_function_via_cgd_class = cgd.graph_full_function(thcf)
graph_interval_function_via_cgd_class = cgd.graph_interval_function(thcf)
calc_cgd = cgd.calculate(thcf, thcf_partial1, thcf_partial2)
graph_full_cgd = cgd.graph_full_coordinate(thcf, thcf_partial1, thcf_partial2)
graph_partial_cgd = cgd.graph_partial_coordinate(thcf, thcf_partial1,
                                                 thcf_partial2)

# Steepest Gradient Descent Usage Examples:

sgd = SteepestGradientDescent([5, 5], [5, 5], [5, 5],
                              1e-3, 1e-6, 1, 10000)
graph_full_function_via_sgd_class = sgd.graph_full_function(thcf)
graph_interval_function_via_sgd_class = sgd.graph_interval_function(thcf)
calc_sgd = sgd.calculate(thcf, thcf_partial1, thcf_partial2)
graph_full_sgd = sgd.graph_full_steepest(thcf, thcf_partial1, thcf_partial2)
graph_partial_sgd = sgd.graph_partial_steepest(thcf, thcf_partial1,
                                               thcf_partial2)
