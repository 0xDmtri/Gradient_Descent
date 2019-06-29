import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class SteepestGradientDescent:
    def __init__(self, initial_x, current_x, prev_x,
                 lr, tolerance, step, max_iters):
        """ Initiate variables for function minimisation:
                1) Initial X coordinate.
                2) Current X coordinate.
                3) Learning Rate (lr).
                4) Tolerance level.
                5) Previous step size.
                6) Maximum number of iterations.

            At the moment algorithm works correctly with the given tolerance
            level and number of maximum iterations.

            However, decreasing the tolerance and increaseing the maximum
            number of iteration is advised to handle larger scope of
            functions to miminise.

            Computational time is expected to increawse in response to
            the proposed model alterations.

        """
        self.initial_x = initial_x
        self.current_x = current_x
        self.prev_x = prev_x
        self.lr = lr
        self.tolerance = tolerance
        self.step = step
        self.max_iters = max_iters
        self.iter = 0
        self.x_path = pd.DataFrame(index=range(self.max_iters),
                                   columns=range(2))

    def calculate(self, equation, partial1, partial2):
        """ Minimise the selected function via Steepest Descent Algorithm.

            Arguments:
                1) Function to minimise.
                2) First partial derivative of the function.
                3) Second partial derivative of the function.
        """
        self.x_path[0][self.iter] = float(self.initial_x[0])
        self.x_path[1][self.iter] = float(self.initial_x[1])
        while (self.step > self.tolerance) and (self.iter < self.max_iters):
            self.prev_x = self.current_x.copy()
            self.current_x[0] = float(self.current_x[0]) - self.lr * float(
                partial1(self.prev_x))
            self.current_x[1] = float(self.current_x[1]) - self.lr * float(
                partial2(self.prev_x))
            self.step = abs(float(equation(self.prev_x))
                            - float(equation(self.current_x)))
            self.iter += 1
            self.x_path[0][self.iter] = float(self.current_x[0])
            self.x_path[1][self.iter] = float(self.current_x[1])
            number_of_iters = str("Number of iterations before achieving the\
                                   minimum: %s" % self.iter)
            coordinate = str("Coordinates of the minimum: %s" % self.current_x)
        return number_of_iters, coordinate

    @staticmethod
    def graph_full_function(equation):
        """ Graph full surface of the selected function.

            Arguments:
                1) Function to plot.
        """
        x1 = [[], []]
        x1[0] = np.arange(-5, 5, 0.1)
        x1[1] = np.arange(-5, 5, 0.1)
        x1[0], x1[1] = np.meshgrid(x1[0], x1[1])
        y1 = equation(x1)

        fig1 = plt.figure(figsize=(10, 10))
        ax1 = fig1.add_subplot(111, projection='3d')
        full1 = ax1.plot_surface(x1[0], x1[1], y1, cmap=cm.seismic,
                                 linewidth=0, antialiased=False)
        ax1.set_xlabel('Value of the first independent variable', fontsize=10)
        ax1.set_ylabel('Value of the second independent variable', fontsize=10)
        ax1.set_zlabel('Value of the dependent function', fontsize=10)
        fig1.colorbar(full1, shrink=0.5, aspect=10)

    @staticmethod
    def graph_interval_function(equation):
        """ Graph partial surface of the selected function.

            Arguments:
                1) Function to plot.
        """
        x2 = [[], []]
        x2[0] = np.arange(-2, 2, 0.1)
        x2[1] = np.arange(-2, 2, 0.1)
        x2[0], x2[1] = np.meshgrid(x2[0], x2[1])
        y2 = equation(x2)

        fig2 = plt.figure(figsize=(10, 10))
        ax2 = fig2.add_subplot(111, projection='3d')
        full2 = ax2.plot_surface(x2[0], x2[1], y2, cmap=cm.seismic,
                                 linewidth=0, antialiased=False)
        ax2.set_xlabel('Value of the first independent variable', fontsize=10)
        ax2.set_ylabel('Value of the second independent variable', fontsize=10)
        ax2.set_zlabel('Value of the dependent function', fontsize=10)
        fig2.colorbar(full2, shrink=0.5, aspect=10)

    def graph_full_steepest(self, equation, partial1, partial2):
        """ Graph full path of the XY coordinate.

            Arguments:
                1) Function to plot.
                2) First partial derivative of the function.
                3) Second partial derivative of the function.
        """
        self.calculate(equation, partial1, partial2)
        x_path_local = self.x_path.drop(labels=None, axis=0, index=range(
                                        self.iter+1, self.max_iters))
        x_path_grid = [[], []]
        x_path_grid[0] = x_path_local[0].values
        x_path_grid[1] = x_path_local[1].values
        y3 = equation(x_path_grid)

        fig3 = plt.figure(figsize=(10, 10))
        ax3 = fig3.gca(projection='3d')
        ax3.plot(x_path_grid[0], x_path_grid[1], y3, label='Steepest Descent')
        ax3.set_xlabel('Value of the first independent variable', fontsize=10)
        ax3.set_ylabel('Value of the second independent variable', fontsize=10)
        ax3.set_zlabel('Value of the dependent function', fontsize=10)
        handles, labels = ax3.get_legend_handles_labels()
        ax3.legend(handles, labels)

    def graph_partial_steepest(self, equation, partial1, partial2):
        """ Graph partial path of the XY coordinate.

            Arguments:
                1) Function to plot.
                2) First partial derivative of the function.
                3) Second partial derivative of the function.
        """
        self.calculate(equation, partial1, partial2)
        x_path_partial = self.x_path.drop(labels=None, axis=0,
                                          index=range(0, 50))
        x_path_grid = [[], []]
        x_path_grid[0] = x_path_partial[0].values
        x_path_grid[1] = x_path_partial[1].values
        y4 = equation(x_path_grid)
        fig4 = plt.figure(figsize=(10, 10))
        ax4 = fig4.gca(projection='3d')
        ax4.plot(x_path_grid[0], x_path_grid[1], y4, label='Steepest Descent')
        ax4.set_xlabel('Value of the first independent variable', fontsize=10)
        ax4.set_ylabel('Value of the second independent variable', fontsize=10)
        ax4.set_zlabel('Value of the dependent function', fontsize=10)
        handles, labels = ax4.get_legend_handles_labels()
        ax4.legend(handles, labels)
