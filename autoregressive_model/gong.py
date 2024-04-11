import numpy as np
import scipy.io.wavfile as wav
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def ar_fit_model(y: np.ndarray, p: int) -> np.ndarray:
    y_stack = np.array([np.flip(y[i: i + p]) for i in range(y.shape[0] - p)])
    matrix = np.hstack((np.ones((y_stack.shape[0], 1)), y_stack))
    b = y[p:]
    ret, residuals, rank, singular_values = np.linalg.lstsq(matrix, b, rcond=None)
    return ret


def ar_predict(a: np.ndarray, y0: np.ndarray, N:int) -> np.ndarray:
    p = y0.shape[0]
    y_pred = np.zeros(N)
    y_pred[:p] = y0
    for i in range(N - p):
        iter_matrix = np.hstack((np.array([1]), np.flip(y_pred[i: i + p])))
        y_pred[i + p] = iter_matrix @ np.reshape(a, (a.shape[0], 1))
    return y_pred


def main():
    fs, y = wav.read('gong.wav')
    #y = y.copy() / 32767
    y = np.arange(1, 15)
    p = 3  # size of history considered for prediction
    N = len(y)  # length of the sequence
    K = 10000  # visualize just K first elements
    a = ar_fit_model(y, p)
    y0 = y[:p]
    y_pred = ar_predict(a, y0, N)
    wav.write('gong_predicted.wav', fs, y_pred)
    plt.plot(y[:K], 'b', label='original')
    plt.plot(y_pred[:K], 'r', label='AR model')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()


