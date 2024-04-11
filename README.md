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

