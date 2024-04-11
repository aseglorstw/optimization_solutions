### PCA: Motion Capture Solution

In this solution, I'll be using PCA (Principal Component Analysis) to fit points in space with a linear or affine subspace and minimize the error. 

#### Objective:
The main focus will be on approximating motion capture data with lower-dimensional affine subspaces. This involves finding the principal components of the data and representing them with a linear or affine subspace using PCA. 

#### Methodology:
The goal is to reduce data volume while retaining the essential characteristics of the motion. I'll implement functions to fit the data to subspace representations, visualize the results, and analyze the compression error across different dimensions of the subspace.

<p>
  <img src="https://ltdfoto.ru/images/2024/04/11/85ef1759-95e3-4162-b961-33d521e3d131.jpg">
</p>

### Autoregressive model: Gong Solution

#### Objective:

In this solution, our goal is to estimate the parameters of an autoregressive model for a time series data sequence. The autoregressive model predicts the value of the current data point based on its previous values. We aim to minimize the sum of squared differences between the predicted values and the actual data points.

#### Methodology:

To achieve this, we'll implement two functions:

1. `ar_fit_model(y, p)`: This function takes a column vector `y` as the measured sample, a scalar `p` as the order of the model, and returns a column vector `a` representing the estimated parameters. We'll formulate this task as the minimization of the function ||Ma - b||^2.

2. `ar_predict(a, y0, N)`: This function uses the estimated parameters `a` to predict future values of the time series. It takes the parameter vector `a`, an initial subsequence `y0` of length `p`, and the desired length `N` of the predicted sequence. It returns a column vector `y_pred` containing the predicted sequence.

We'll provide templates and scripts for testing in both MATLAB and Python. The resulting plot for the 'gong' dataset should resemble the one provided. Additionally, we'll devise simple test cases for further evaluation, such as constant and linear regressions (p = 0, 1).

<p align="center">
  <img src="https://cw.fel.cvut.cz/wiki/_media/courses/b0b33opt/cviceni/hw/lsq2/gong.png">
</p>


### Prokládání bodů kružnicí Solution
#### Objective:
The task is to find the best-fitting circle for a given set of points in a 2D plane. This involves solving two subproblems, each formalizing the notion of "best-fitting" differently.

#### Methodology:
Minimization of Algebraic Distance:
Formulate the problem as finding a circle that minimizes the sum of squared distances between the points and the circle. This involves approximating the circle using the equation of a conic section and solving the resulting optimization problem.
Robust Fitting using RANSAC Algorithm:
Implement the RANSAC (Random Sample Consensus) algorithm, which robustly estimates the parameters of the circle by iteratively fitting the circle to randomly sampled subsets of the data and selecting the model with the most inliers, i.e., points that are consistent with the model.
Tasks:
Implement the following functions:
1. `d = dist(X, x0, y0, r)`: Computes the distances of points from the circle defined by center (x0, y0) and radius r.
2. `[x0 y0 r] = quad_to_center(d, e, f)`: Converts the representation of the circle from the equation of a conic section to the standard circle equation.
3. `[d e f] = fit_circle_nhom(X)`: Fits the circle using a non-homogeneous approach.
4. `[d e f] = fit_circle_hom(X)`: Fits the circle using a homogeneous approach.
5. `[x0 y0 r] = fit_circle_ransac(X, num_iter, threshold)`: Implements the RANSAC algorithm to robustly fit the circle to the data.
These functions provide different strategies to find the best-fitting circle for the given points, addressing various aspects such as algebraic distance minimization and robustness to outliers.

<p align="center">
  <img src="https://cw.fel.cvut.cz/wiki/_media/courses/b0b33opt/cviceni/hw/kruznice_lin/circle-fit-nhom.svg?w=300&tok=cf4b28" width="30%">
  <img src="https://cw.fel.cvut.cz/wiki/_media/courses/b0b33opt/cviceni/hw/kruznice_lin/circle-fit-hom.svg?w=300&tok=295d15" width="30%">  
  <img src="https://cw.fel.cvut.cz/wiki/_media/courses/b0b33opt/cviceni/hw/kruznice_lin/circle-fit-ransac.svg?w=300&tok=6e417b" width="30%">  
</p>

### Minimaxní lineární regrese Solution

#### Objective:

The task is to find the affine function that best approximates a given set of points in a way that minimizes the maximum absolute deviation from the function to the points.

#### Methodology:

1. **Linear Programming Formulation:** 
   - Convert the problem into a linear program to find the parameters of the affine function that minimize the maximum absolute deviation.

2. **Implementation of Functions:**
   - Implement the function `minimaxfit(x, y)` to find the parameters `a` and `b` of the affine function, along with the minimum value `r` of the maximum absolute deviation criterion.
   - Implement the function `plotline(x, y, a, b, r)` to visualize the points and the found band, specifically for the case when `n = 1`.

#### Tasks:

- Convert the problem into a linear program.
- Implement the function `minimaxfit(x, y)` to find the parameters of the affine function.
- Implement the function `plotline(x, y, a, b, r)` to visualize the points and the found band, specifically for the case when `n = 1`.

<p align="center">
  <img src="https://cw.fel.cvut.cz/wiki/_media/courses/b0b33opt/cviceni/hw/lp1/linefit/body2.svg?w=400&tok=45a329">
</p>

