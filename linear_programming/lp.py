import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


def vyhra(c, k):
    A_ub = np.array([[-c[0], -c[1], 0, 0, 0, 1],
                     [0, -c[1], -c[2], -c[3], 0, 1],
                     [0, 0, 0, -c[3], -c[4], 1]])
    b_ub = np.array([0, 0, 0])
    A_eq = np.array([[1, 1, 1, 1, 1, 0]])
    b_eq = np.array([k])
    coefs = np.array([0, 0, 0, 0, 0, -1])
    bounds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None))
    res = scipy.optimize.linprog(c=coefs, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    return res.x[:-1]


def vyhra2(c, k, m):
    A_ub = np.array([[-c[0], 0, 0, 1],
                     [0, -c[1], 0, 1],
                     [0, 0, -c[2], 1]])
    b_ub = np.array([0, 0, 0])
    A_eq = np.array([[1, 1, 1, 0]])
    b_eq = np.array([k])
    coeffs = np.array([0, 0, 0, -1])
    bounds = ((m, None), (m, None), (m, None), (0, None))
    res = scipy.optimize.linprog(c=coeffs, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    return res.x[:-1]


def minimaxfit(x, y):
    A_ub_up = np.hstack((x.T, np.ones((y.shape[1], 1)), -np.ones((y.shape[1], 1))))
    A_ub_down = np.hstack((-x.T, -np.ones((y.shape[1], 1)), -np.ones((y.shape[1], 1))))
    A_ub = np.vstack((A_ub_up, A_ub_down))
    b_ub = np.vstack((y.T, -y.T)).T[0]
    coeffs = np.hstack((np.zeros((1, x.shape[0] + 1)), np.array([[1]])))[0]
    bounds = [(None, None)] * (x.shape[0] + 1)
    bounds.append((0, None))
    res = scipy.optimize.linprog(c=coeffs, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    a = res.x[:x.shape[0]]
    b = res.x[x.shape[0]]
    r = res.x[x.shape[0] + 1]
    return np.reshape(np.array(a.T), (x.shape[0], 1)).squeeze(), b, r

def plotline(x, y, a, b, r):
    x_line = range(min(x[0]) - 1, max(x[0]) + 1)
    plt.plot(x, y, 'x', color='black')
    plt.plot(x_line, a * x_line + b, 'r-', color='green')
    plt.plot(x_line, a * x_line + b + r, 'r-', color='red')
    plt.plot(x_line, a * x_line + b - r, 'r-', color='red')
    plt.show()


if __name__ == "__main__":
    # x = np.random.uniform(low=0, high=20, size=(5, 5))  # Генерация случайной матрицы x
    # y = np.random.uniform(low=0, high=20, size=(1, 5))
    # print(x)
    # print(y)
    x = np.array([[1, 2, 3, 3, 2], [4, 1, 2, 5, 6], [7, 8, 9, -5, 7]])
    y = np.array([[7, 4, 1, 2, 5]])
    a, b, r = minimaxfit(x, y)
    print(a)
    print(b)
    print(r)

