import scipy.io as sio
import numpy as np
from math import pi
from matplotlib import pyplot as plt



def quad_to_center(d, e, f):
    x0 = -1/2 * d
    y0 = -1/2 * e
    r = -(f - x0 ** 2 - y0**2)
    return x0, y0, r


def fit_circle_nhom(X):
    X = np.array(X)
    A = np.hstack((X, np.ones((X.shape[0], 1))))
    x, y = X[:, 0], X[:, 1]
    b = -x ** 2 - y ** 2
    x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
    return x[0], x[1], x[2]


def fit_circle_hom(X):
    X = np.array(X)
    x, y = X[:, 0], X[:, 1]
    sum = np.reshape(np.array(x ** 2 + y ** 2), (X.shape[0], 1))
    A = np.hstack((sum, X, np.ones((X.shape[0], 1))))
    eigenvalues, eigenvectors = np.linalg.eigh(A.T @ A)
    a, d, e, f = eigenvectors.T[np.argmin(eigenvalues)]
    d = d / a
    e = e / a
    f = f / a
    return d, e, f


def dist(X, x0, y0, r):
    points_np = np.array(X)
    x1, y1 = points_np[:, 0], points_np[:, 1]
    distances = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) - r
    return distances


def fit_circle_ransac(X, num_iter, threshold):
    X = np.array(X)
    opt_solution = 0
    opt_arg = None
    for _ in range(num_iter + 1):
        random_indices = np.random.choice(len(X), 3, replace=False)
        random_points = X[random_indices]
        d, e, f = fit_circle_hom(random_points)
        x0, y0, r = quad_to_center(d, e, f)
        dists = dist(X, x0, y0, r)
        number_inliers = np.sum(abs(dists) < threshold)
        if number_inliers > opt_solution:
            opt_solution = number_inliers
            opt_arg = (x0, y0, r)
    return opt_arg


def plot_circle(x0,y0,r, color, label):
    t = np.arange(0,2*pi,0.01)
    X = x0 + r*np.cos(t)
    Y = y0 + r*np.sin(t)
    plt.plot(X,Y, color=color, label=label)


if(__name__ == '__main__'):
    data = sio.loadmat('data.mat')
    X = data['X'] # only inliers
    A = data['A'] # X + outliers

    def_nh = fit_circle_nhom(X)
    x0y0r_nh = quad_to_center(*def_nh)
    dnh = dist(X, *x0y0r_nh)
    def_h = fit_circle_hom(X)
    x0y0r_h = quad_to_center(*def_h)
    dh = dist(X, *x0y0r_h)

    results = {'def_nh':def_nh, 'def_h':def_h,
               'x0y0r_nh' : x0y0r_nh, 'x0y0r_h': x0y0r_nh,
               'dnh': dnh, 'dh':dh}

    GT = sio.loadmat('GT.mat')
    for key in results:
        print('max difference',  np.amax(np.abs(results[key] - GT[key])), 'in', key)


    x = fit_circle_ransac(A, 2000, 0.1)

    plt.figure(1)
    plt.subplot(121)
    plt.scatter(X[:,0], X[:,1], marker='.', s=3)
    plot_circle(*x0y0r_h, 'r', 'hom')
    plot_circle(*x0y0r_nh, 'b', 'nhom')
    plt.legend()
    plt.axis('equal')
    plt.subplot(122)
    plt.scatter(A[:,0], A[:,1], marker='.', s=2)
    plot_circle(*x, 'y', 'ransac')
    plt.legend()
    plt.axis('equal')
    plt.show()

