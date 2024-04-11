import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np


def fit_wages(times, wages):
    evaluated_phi1 = np.reshape(times, (times.shape[0], 1))
    evaluated_phi2 = np.ones((times.shape[0], 1))
    matrix = np.hstack((evaluated_phi1, evaluated_phi2))
    x, residuals, rank, singular_values = np.linalg.lstsq(matrix, wages, rcond=None)
    return x[1], x[0]


def quarter2_2009(x):
    m, k = x
    return k * 2009.25 + m


def show_result_wages(times, wages, x):
    m, k = x
    coord_x = np.arange(2000, 2011)
    coord_y = k * coord_x + m
    plt.plot(coord_x, coord_y, label='Linear Fit')
    plt.scatter(times, wages, color='red', label='Data Points')
    plt.xlabel('Year')
    plt.ylabel('Wages')
    plt.legend()
    plt.show()


def fit_temps(times, temperatures, omega):
    evaluated_phi1 = np.ones((times.shape[0], 1))
    evaluated_phi2 = np.reshape(times, (times.shape[0], 1))
    evaluated_phi3 = np.reshape(np.sin(omega * times), (times.shape[0], 1))
    evaluated_phi4 = np.reshape(np.cos(omega * times), (times.shape[0], 1))
    matrix = np.hstack((evaluated_phi1, evaluated_phi2, evaluated_phi3, evaluated_phi4))
    y = np.reshape(temperatures, (temperatures.shape[0], 1))
    x, residuals, rank, singular_values = np.linalg.lstsq(matrix, y, rcond=None)
    x1, x2, x3, x4 = x.flatten()
    return x1, x2, x3, x4


def show_result_temperatures(times, temperatures, omega, x):
    c1, c2, c3, c4 = x
    coord_x = np.arange(0, 1200)
    coord_y = c1 + c2 * coord_x + c3 * np.sin(coord_x * omega) + c4 * np.cos(coord_x * omega)
    plt.plot(coord_x, coord_y, label='Linear Fit')
    plt.scatter(times, temperatures, color='red', label='Data Points')
    plt.xlabel('Year')
    plt.ylabel('Temperatures')
    plt.legend()
    plt.show()


def main():
    data_wages = np.loadtxt("mzdy.txt")
    times_wages = data_wages[:, 0]
    wages = data_wages[:, 1]
    x_wages = fit_wages(times_wages, wages)
    wage_2009_5 = quarter2_2009(x_wages)
    show_result_wages(times_wages, wages, x_wages)
    m, k = x_wages
    print(f"\nApproximation of the wages function using the function y = kx + m, \n "
          f"where k = {k}, m = {m}.")
    print(f"Wage in 2009.25 is {wage_2009_5}.\n")

    data_temperatures = np.loadtxt("teplota.txt")
    times_temperatures = data_temperatures[:, 0]
    temperatures = data_temperatures[:, 1]
    omega = 2 * math.pi / 365
    x_temperatures = fit_temps(times_temperatures, temperatures, omega)
    print(x_temperatures)
    show_result_temperatures(times_temperatures, temperatures, omega, x_temperatures)
    print(f"Approximation of the temperature function using the function y = c1 + c2*x + c3*sin(wx) + c4*cos(wx), \n "
           f"where c1 = {x_temperatures[0]}, c2 = {x_temperatures[1]}, c3 = {x_temperatures[2]}",
           f"c4 = {x_temperatures[3]} and w = {omega}.")


if __name__ == "__main__":
    main()

