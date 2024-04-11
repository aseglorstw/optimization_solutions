from typing import Tuple

import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import scipy.io as sio
from matplotlib import animation
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def playmotion(conn, A, B = None):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.axis('off')

    conns = [x[x!=41] for x in np.split(conn, np.where(conn==41)[0]) if len(x[x!=41])]

    macAs = []
    macBs = []

    m, n = A.shape

    if B is not None:
        B = B.reshape(3, m//3, n, order='F')
        for conn in conns:
            macBs.append(ax.plot(B[0, conn, 0], B[1, conn, 0], B[2, conn, 0], marker='o', color='r')[0])

    A = A.reshape(3, m//3, n, order='F')
    for conn in conns:
        macAs.append(ax.plot(A[0, conn, 0], A[1, conn, 0], A[2, conn, 0], marker='o', color='b')[0])

    fig.legend(handles=[mpatches.Patch(color='red', label='approximation'), mpatches.Patch(color='blue', label='GT')])
    set_axes_equal(ax)

    def update_points(i, A, B, macAs, macBs, conn):
        for conn, macA in zip(conns, macAs):
            macA.set_data(np.array(A[:2,conn,i]))
            macA.set_3d_properties(A[2,conn,i], 'z')
        for conn, macB in zip(conns, macBs):
            macB.set_data(np.array(B[:2,conn,i]))
            macB.set_3d_properties(B[2,conn,i], 'z')
        return macAs + macBs
    
    ani = animation.FuncAnimation(fig, update_points, n, fargs=(A, B, macAs, macBs, conns), interval=1)
    plt.show()


def fitlin(A: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(A @ A.T)
    matrix_U = eigenvectors[:, -k:]
    matrix_B = matrix_U @ matrix_U.T @ A
    matrix_C = matrix_U.T @ matrix_B
    return matrix_U, matrix_C


def fitaff(A: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centre_of_gravity = np.reshape(1/A.shape[1] * np.sum(A, axis=1), (A.shape[0], 1))
    matrix_U_around_origin, matrix_C = fitlin(A - centre_of_gravity, k)
    return matrix_U_around_origin, matrix_C, centre_of_gravity


def erraff(A: np.ndarray) -> np.ndarray:
    result = np.zeros(A.shape[0] + 1)
    centre_of_gravity = np.reshape(1 / A.shape[1] * np.sum(A, axis=1), (A.shape[0], 1))
    points_around_origin = A - centre_of_gravity
    eigenvalues, eigenvectors = np.linalg.eigh(points_around_origin @ points_around_origin.T)
    reversed_eigenvalues = eigenvalues[::-1]
    for k in range(1, A.shape[0] + 1):
        result[k] = sum(reversed_eigenvalues[k:])
    return result[1:]


def drawfitline(A: np.ndarray) -> None:
    matrix_U, matrix_C, b0 = fitaff(A, 1)
    B = matrix_U @ matrix_C + b0.reshape(-1, 1)
    m = (matrix_U[1] / matrix_U[0])[0]
    max_coord_x = max(A[0]) if max(A[0]) > max(B[0])  else max(B[0])
    min_coord_x = min(A[0]) if min(A[0]) < min(B[0]) else min(B[0])
    coord_line_x = np.arange(min_coord_x, max_coord_x + 1)
    coord_line_y = m * (coord_line_x - b0[0]) + b0[1]
    plt.plot(coord_line_x, coord_line_y, color='green', label='Line')
    plt.scatter(A[0], A[1], color='red', marker='+')
    for i in range(A.shape[1]):
        plt.plot([A[0, i], B[0, i]], [A[1, i], B[1, i]], color='red')
    plt.subplot(221)
    plt.plot(A)
    plt.title('drawfitline')



def plottraj2(C: np.ndarray) -> None:
    plt.plot(C[0], C[1], marker='o', linestyle='-')
    plt.subplot(222)
    plt.plot(C)
    plt.title('plottraj2')


def main():
    A = sio.loadmat('data/line.mat')['A']
    drawfitline(A)

    conn = np.loadtxt('data/connected_points.txt', comments='%', dtype=int) - 1
    filename = 'makarena1.txt'  # see the data folder and try more examples
    A = np.loadtxt('data/' + filename).T
    k = 2  # dimension of affine approximation

    U, C, b0 = fitaff(A, k)
    B = U @ C + b0.reshape(-1, 1)

    plottraj2(C[:2])

    plt.subplot(212)
    plt.semilogy(erraff(A))
    plt.xlabel('dimension')
    plt.ylabel('error, log scale')
    plt.title('Error of affine approximation \n for motion capture')

    plt.tight_layout()
    playmotion(conn, A, B)


if __name__ == '__main__':
    main()
