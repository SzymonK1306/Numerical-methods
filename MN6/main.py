import numpy as np
import math
from scipy.optimize import fmin
import matplotlib.pyplot as plt

data = np.loadtxt("lab_6_dane/data11.txt")

# data from file
y = np.array(np.copy(data[:, 1]))
t = np.array(np.copy(data[:, 0]))

def cost_function(params):

    k, tau, zeta, tau_z = params

    omega_n = 1 / tau
    omega_zero = (math.sqrt(1 - (zeta ** 2))) / tau

    g_t = impulse_response_g_t(omega_n, zeta, omega_zero, t)
    h_t = step_response_h_t(omega_n, zeta, omega_zero, t)

    ys = np.mat(k * (tau_z * g_t + h_t))

    yy = np.mat(y)
    return ((ys - yy) * (ys - yy).T).item(0)


def impulse_response_g_t(wn, zeta, w0, t):
    return (wn / np.sqrt(1 - zeta ** 2)) * np.multiply(np.exp(-zeta * wn * t),
                                                       np.sin(w0 * t))


def step_response_h_t(wn, zeta, w0, t):
    return 1 - np.multiply((np.exp(-zeta * wn * t)), np.cos(w0 * t) + (
                zeta * np.sin(w0 * t) / np.sqrt(1 - zeta ** 2)))


def step_response(params):
    k, tau, zeta, tau_z = params

    omega_n = 1 / tau
    omega_zero = (math.sqrt(1 - (zeta ** 2))) / tau

    g_t = impulse_response_g_t(omega_n, zeta, omega_zero, t)
    h_t = step_response_h_t(omega_n, zeta, omega_zero, t)

    ys = k * (tau_z * g_t + h_t)

    return ys


initial_params = np.array([0, 0, 0, 0])
params_out = fmin(cost_function, initial_params)

print(params_out)

# Plotting

plt.plot(t, y, 'r', t, step_response(params_out), 'g')
plt.legend(["data", "approximation"])

plt.grid()

plt.show()