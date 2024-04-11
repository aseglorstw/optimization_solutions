import numpy as np
import scipy.io as sio
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from typing import Callable


def LM_iter(X: np.array, x0: float, y0: float, r0: float, mu: float) -> tuple:
    points_np = np.array(X)
    x1, y1 = points_np[:, 0], points_np[:, 1]
    jacob_x = np.array([((x0 - x1) / np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))])
    jacob_y = np.array([((y0 - y1) / np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))])
    jacob_r = -np.ones((1, jacob_x.shape[1]))
    jacob = np.vstack((jacob_x, jacob_y, jacob_r)).T
    dists_g = np.array(dist(X, x0, y0, r0))
    g_func = np.reshape(dists_g, (dists_g.shape[0], 1))
    vx, vy, vr = -np.linalg.inv(jacob.T @ jacob + mu * np.eye(jacob.T.shape[0])) @ jacob.T @ g_func
    f = get_objective_function(X)
    old_value = f([x0, y0, r0])
    new_value = f([x0 + vx[0], y0 + vy[0], r0 + vr[0]])
    is_success = new_value < old_value
    print(is_success, new_value, old_value)
    return x0 + vx[0], y0 + vy[0], r0 + vr[0], is_success


def GN_iter(X: np.array, x0: float, y0: float, r0: float) -> tuple:
    points_np = np.array(X)
    x1, y1 = points_np[:, 0], points_np[:, 1]
    jacob_x = np.array([((x0 - x1) / np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))])
    jacob_y = np.array([((y0 - y1) / np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))])
    jacob_r = -np.ones((1, jacob_x.shape[1]))
    jacob = np.vstack((jacob_x, jacob_y, jacob_r)).T
    dists_g = np.array(dist(X, x0, y0, r0))
    g_func = np.reshape(dists_g, (dists_g.shape[0], 1))
    vx, vy, vr = -np.linalg.inv(jacob.T @ jacob) @ jacob.T @ g_func
    return x0 + vx[0], y0 + vy[0], r0 + vr[0]


def grad_iter(X: np.array, x0: float, y0: float, r0: float, a: float) -> tuple:
        points_np = np.array(X)
        x1, y1 = points_np[:, 0], points_np[:, 1]
        grad_dist_x = 2 * np.sum(dist(X, x0, y0, r0) * ((x0 - x1) / np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)))
        grad_dist_y = 2 * np.sum(dist(X, x0, y0, r0) * ((y0 - y1) / np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)))
        grad_r = -2 * sum(dist(X, x0, y0, r0))
        return x0 - a * grad_dist_x, y0 - a * grad_dist_y, r0 - a * grad_r


def dist(X, x0, y0, r):
    points_np = np.array(X)
    x1, y1 = points_np[:, 0], points_np[:, 1]
    distances = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) - r
    return np.array(distances)


def get_objective_function(X: np.array) -> Callable:
    return lambda c: np.sum(dist(X, c[0], c[1], c[2]) ** 2)

def plot_circle(ax, x0, y0, r, color):
    theta = np.arange(0, 2*np.pi, 0.01)
    xs = r*np.cos(theta) + x0
    ys = r*np.sin(theta) + y0
    ax.plot(xs,ys, color)


if __name__ == '__main__':

    use_user_input_points = False
    if(use_user_input_points):
        try:
            X = sio.loadmat('points.mat')['X']
        except:
            num_points = 10
            plt.title('Select ' + str(num_points) + ' X')
            plt.draw()
            X = np.asarray(plt.ginput(num_points))
            sio.savemat('points.mat', {'X':X})
    else:
        x0 = 5
        y0 = 6
        r = 7
        theta = np.random.rand(150)*2*np.pi
        n = len(theta)
        X = np.zeros((n,2))
        X[:,0] = r*np.cos(theta) + x0
        X[:,1] = r*np.sin(theta) + y0
        X = X + np.random.randn(n,2)*3

    # set initial parameters to r=1 and x0,y0 = mean(X)
    params = np.ones((3,3))
    params[:,:2] = np.mean(X,0)

    # calculate initiall ssd. assumes same init for all methods
    d = np.sum(dist(X, *params[0])**2)
    dists = [[d],[d],[d]]
    mu = 1
    num_iter = 10
    # step size
    a = 0.01

    res = minimize(get_objective_function(X), x0=params[0])
    print('get_objective_function:')
    print(' objective', res.fun)
    print(' circle params x0, y0, r:', res.x)
    print()
    for i in range(num_iter):
        if(i):
            # after timeout sec it will continue by itself
            # otherwise will wait for button press
            timeout = 0.1
            plt.waitforbuttonpress(timeout)

            # if you close the figure, program should stop
            if(not plt.get_fignums()):
                break

        plt.clf()
        ax = plt.subplot(2,2,1)
        ax.title.set_text('LM')
        ax.plot(X[:,0], X[:,1], '.')
        plot_circle(ax, *params[0], 'r')
        x,y,r,succ = LM_iter(X, *params[0], mu)
        dists[0].append(np.sum(dist(X, x,y,r)**2))
        if(succ):
            params[0] = np.array([x,y,r])
            mu /= 3
        else:
            mu *= 2
        ax.axis('equal')

        ax = plt.subplot(2,2,2)
        ax.title.set_text('GN')
        ax.plot(X[:,0], X[:,1], '.')
        plot_circle(ax, *params[1], 'g')
        x,y,r = GN_iter(X, *params[1])
        dists[1].append(np.sum(dist(X, x,y,r)**2))
        params[1] = np.array([x,y,r])
        ax.axis('equal')


        ax = plt.subplot(2,2,3)
        ax.title.set_text('grad')
        ax.plot(X[:,0], X[:,1], '.')
        plot_circle(ax, *params[2],'k')
        x,y,r = grad_iter(X, *params[2], a)
        new_dist = np.sum(dist(X, x,y,r)**2)
        if(new_dist > dists[2][-1]):
            a /= 2
            new_dist =  dists[2][-1]
        else:
            a *= 2
            params[2] = np.array([x,y,r])
        dists[2].append(new_dist)
        ax.axis('equal')

        ax = plt.subplot(4,2,6)
        ax.title.set_text('error progression')
        ax.plot(dists[0], 'r-')
        ax.plot(dists[1], 'g--')
        ax.plot(dists[2], 'k-.')
        ax.ticklabel_format(useOffset=False)

        ax = plt.subplot(4,2,8)
        ax.title.set_text('error of last 5 iters')
        ax.plot(dists[0][-5:], 'r-' )
        ax.plot(dists[1][-5:], 'g--' )
        ax.plot(dists[2][-5:], 'k-.' )
        ax.ticklabel_format(useOffset=False)

        plt.tight_layout()
        plt.draw()
        # print('Iter %d:' %(i+1))
        # print('LM: f=%f, success=%d, mu=%f' %(dists[0][-1], succ, mu))
        # print('GN: f=%f' %dists[1][-1] )
        # print('GM: f=%f, a: %a' %(dists[2][-1], a))
        # print()
    plt.show()