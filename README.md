### Part 1: Gradient Descent ###

## Importance:

Gradient descent is a fundamental optimisation algorithm for machine
learning models. Such importance is derived from algorithm versatility
as it can be applied for virtually any model with no limitations for
number of functional spaces, simplicity/cheapness of computation
comparative to algebraic methods is the second reason.

## Working principle:

Gradient descent general principle is to minimizes the cost function
which represents error in predictions which in case of linear or
logistic regression is mean squared error (MSE) or mean absolute error
(MAE). The

Plain Vanilla gradient works by a principle of steepest descent. Partial
derivatives of cost function with respect to (w.r.t) variables are taken
to obtain a gradient for the current state of function. Then algorithm
takes a step-in opposite direction to the gradient. In the first
coordinate step in the opposite direction of first variable partial
w.r.t first variable and second step in the opposite direction to the
partial derivative w.r.t second variable, simultaneously. The size of
the taken step is determined by the learning rate. Procedure is repeated
until local or global minimum is reached.

## Plain Vanilla Testing:

For testing the selected objective function is Tree-Hump Camel function.
It depends on two variables[^1]; valley shaped (fig.1) and multiple
local minima are present (fig.2). Global minimum is located at \[0, 0\].

In accordance with aforementioned procedure, optimization was conducted
at learning rates (lr) of 0.01, 0.001 and 0.0001(table 1). The optimal
rate 0.001 brings function to global minima after 6043 iterations (fig
3,4). At 0.01lr algorithm overshoots and the function value explodes
after the fourth iteration, 0.001lr (fig 5,6) on the other hand takes
52869 iterations, yet precision suffers significantly as final
coordinate significantly diverges from global minimum \[0,0\] (table 1).

## Modifications:

# Coordinate Descent:

Coordinate Descent(CD)is a simple algorithm that works in similar
fashion to gradient descent. In multivariate case, CD minimises function
in direction of one variable at a time. Therefore, adjusting
sequentially w.r.t one of the variables function moves towards the
minimum as can be seen in fig7, 8. The algorithm, however overshoots the
gloabal minimum overshoots the global minimum as the partial derivative
depends on the second function (table 1).

# Nesterov's accelerated gradient descent:

Nesterov's accelerated gradient descent is a modification of the
momentum gradient descent that takes into the account the gradient at
the partially updated coordinate. Nesterov's descent is a faster
minimisation algorithm compared to the steepest descent, however, it is
sensitive not only to choose of the learning rate but to the momentum
parameter as well. Across the board the inclusion of momentum increases
the speed and the accuracy of the minimisation algorithm. The best
choice of parameters is 0.001 learning rate and 0.4 momentum. The
algorithm with these parameters takes 3938 iterations to reach the
global minimum (fig 9, 10). The particularly indicative case is the
Nesterov's algorithm with momentum 0.5 and learning rate 0.001 (fig
11,12). The momentum is too large, the algorithm overshoots and gets
stuck at the local minima of 0.298638455185163.

  Table 1: Gradient Descent results.                                                                                                    
  ----------------------------------------- --------------- ---------------- ---------------------- ----------------------------------- ----------------------------------------------------
  Algorithm                                 Learning rate   Momentum value   Minimum achieved       Iteration before reaching minimum   Coordinate
  Coordinate descent                        0.001           N/A              0.298638442236861      50001                               \[-1.7475523458303008, 0.8737761729151781\]
  Coordinate descent                        0.0001          N/A              0.65600527778067       56019                               \[-1.9551126481450094, 0.9780561704546871\]
  Steepest descent                          0.001           N/A              3.14463562151912E-08   6043                                \[-7.621096885624767e-05, 0.00018398955019988027\]
  Steepest descent                          0.0001          N/A              3.15146133300443E-07   52689                               \[-0.00024126151145191, 0.000582457219095918\]
  Nesterov's accelerated gradient descent   0.001           0.25             2.35330469490769E-08   4726                                \[-6.59282387e-05, 1.59164812e-04\]
  Nesterov's accelerated gradient descent   0.0001          0.25             2.3637230802626E-07    40334                               \[-0.00020894, 0.00050444\]
  Nesterov's accelerated gradient descent   0.001           0.4              1.87519919256049E-08   3938                                \[-5.88513093e-05, 1.42079584e-04\]
  Nesterov's accelerated gradient descent   0.0001          0.4              1.89039753298109E-07   32693                               \[-0.00018686, 0.00045111\]
  Nesterov's accelerated gradient descent   0.001           0.5              0.298638455185163      2674                                \[-1.74756367, 0.87389229\]
  Nesterov's accelerated gradient descent   0.0001          0.5              1.5749150797623E-07    27536                               \[-0.00017055, 0.00041175\]

![](media/image1.png){width="3.3464884076990375in"
height="3.3464884076990375in"}![](media/image2.png){width="3.3464884076990375in"
height="3.3464884076990375in"}

![](media/image3.png){width="3.345833333333333in"
height="3.345833333333333in"}![](media/image4.png){width="3.345833333333333in"
height="3.345833333333333in"}

![](media/image5.png){width="3.345833333333333in"
height="3.345833333333333in"}![](media/image6.png){width="3.345833333333333in"
height="3.345833333333333in"}

![](media/image7.png){width="3.345833333333333in"
height="3.345833333333333in"}![](media/image8.png){width="3.345833333333333in"
height="3.345833333333333in"}

![](media/image9.png){width="3.345833333333333in"
height="3.345833333333333in"}![](media/image10.png){width="3.345833333333333in"
height="3.345833333333333in"}

![](media/image11.png){width="3.345833333333333in"
height="3.345833333333333in"}

![](media/image12.png){width="3.345833333333333in"
height="3.345833333333333in"}

![](media/image13.png){width="3.345833333333333in"
height="3.345833333333333in"}![](media/image14.png){width="3.345833333333333in"
height="3.345833333333333in"}

![](media/image15.png){width="3.345833333333333in"
height="3.345833333333333in"}

![](media/image16.png){width="3.345833333333333in"
height="3.345833333333333in"}

![](media/image10.png){width="3.345833333333333in"
height="3.345833333333333in"}![](media/image9.png){width="3.345833333333333in"
height="3.345833333333333in"}

![](media/image17.png){width="3.345833333333333in"
height="3.345833333333333in"}![](media/image18.png){width="3.345833333333333in"
height="3.345833333333333in"}

[^1]: $\mathbf{f(x) = 2}\mathbf{x}_{\mathbf{1}}^{\mathbf{2}}\mathbf{- 1.05}\mathbf{x}_{\mathbf{1}}^{\mathbf{4}}\mathbf{+}\frac{\mathbf{x}_{\mathbf{1}}^{\mathbf{6}}}{\mathbf{6}}\mathbf{+}\mathbf{x}_{\mathbf{1}}\mathbf{x}_{\mathbf{2}}\mathbf{+}\mathbf{x}_{\mathbf{2}}^{\mathbf{2}}$
