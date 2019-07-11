# Part 1: Gradient Descent

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

### Coordinate Descent: ###

Coordinate Descent (CD) is a simple algorithm that works in similar
fashion to gradient descent. In multivariate case, CD minimises function
in direction of one variable at a time. Therefore, adjusting
sequentially w.r.t one of the variables function moves towards the
minimum as can be seen in fig7, 8. The algorithm, however overshoots the
global minimum overshoots the global minimum as the partial derivative
depends on the second function (table 1).

### Nesterov's accelerated gradient descent: ###

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


### Table 1: Gradient Descent Results ###

| Algorithm          | LR     | Momentum | Minimum          | Iterations | Coordinate             |
| ------------------ | ------ | -------- | ---------------- | ---------- | ---------------------- |
| Coordinate Descent | 0.001  | N/A      | 0.29863844223686 | 50001      | [-1.747552, 0.873776]  |
| Coordinate Descent | 0.0001 | N/A      | 0.65600527778067 | 56019      | [-1.955112, 0.978056]  |
| Steepest Descent   | 0.001  | N/A      | 3.1446356215E-08 | 6041       | [-7.62e-05, 0.000184]  |
| Steepest Descent   | 0.0001 | N/A      | 3.1514613330E-07 | 52689      | [-0.000241, 0.000582]  |
| Nesterov Descent   | 0.001  | 0.25     | 2.3533046949E-08 | 4726       | [-6.59e-05, 1.59we-04] |
| Nesterov Descent   | 0.0001 | 0.25     | 2.3637230802E-07 | 40334      | [-0.000208, 0.000504]  |
| Nesterov Descent   | 0.001  | 0.4      | 1.8751991925E-08 | 3938       | [-5.88e-05, 1.421e-04] |
| Nesterov Descent   | 0.0001 | 0.4      | 1.8903975329E-07 | 32693      | [-0.000186, 0.000451]  |
| Nesterov Descent   | 0.001  | 0.5      | 0.29863845518516 | 2674       | [-1.747563, 0.873892]  |
| Nesterov Descent   | 0.0001 | 0.5      | 1.5749150797E-07 | 27536      | [-0.000171, 0.000412]  |




![Figure 1 and Figure 2](https://i.ibb.co/hBRY1J5/1-2.jpg)


![Figure 3 and Figure 3](https://i.ibb.co/VtRDyW8/3-4.jpg)


![Figure 4 and Figure 4](https://i.ibb.co/yXWZp70/5-6.jpg)


![Figure 5 and Figure 6](https://i.ibb.co/ZxrQ2h5/7-8.jpg)


![Figure 7 and Figure 8](https://i.ibb.co/CVDyckc/9-10.jpg)


![Figure 9 and Figure 10](https://i.ibb.co/VtRDyW8/3-4.jpg)


![Figure 11 and Figure 12](https://i.ibb.co/4fSB78K/11-12.jpg)


![Figure 13 and Figure 14](https://i.ibb.co/Bj6223Q/13-14.jpg)


![Figure 15 and Figure 16](https://i.ibb.co/y88pjYB/15-16.jpg)


![Figure 17 and Figure 18](https://i.ibb.co/Y2V52HS/17-18.jpg)


![Figure 19 and Figure 20](https://i.ibb.co/4m6SPD7/19-20.jpg)
