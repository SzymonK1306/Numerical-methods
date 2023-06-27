# Metoda graficzna
# x = -3.655 y = -2.073
# x = -1.956 y = -1.039
# x = 0.196 y = -8.012

import numpy as np
from matplotlib import pyplot as plt
import time


def newton_raphson(xn, yn):
    t0 = time.time_ns()
    for i in range(20):
        f1 = -xn * xn - 5 * xn - 7 - yn
        f2 = -3 * xn * xn + 5 * xn * yn - yn
        # derivatives
        df1dx = -2 * xn - 5
        df1dy = -1

        df2dx = -6 * xn + 5 * yn
        df2dy = 5 * xn - 1

        jacobian = df1dx * df2dy - df1dy * df2dx

        xn = xn - (f1 * df2dy - f2 * df1dy) / jacobian
        yn = yn - (f2 * df1dx - f1 * df2dx) / jacobian

    return xn, yn


x1_vector = np.arange(-5, 1, 0.001)
y1_vector = -np.square(x1_vector) - 5 * x1_vector - 7

x2_vector = np.arange(-5, 0.2, 0.001)
y2_vector = (-3*np.square(x2_vector))/(1-5*x2_vector)
x3_vector = np.arange(0.21, 1, 0.001)
y3_vector = (-3*np.square(x3_vector))/(1-5*x3_vector)

fig, ax = plt.subplots()
plt.plot(x1_vector, y1_vector, 'r')
plt.plot(x2_vector, y2_vector, 'b')
plt.plot(x3_vector, y3_vector, 'b')

# metoda iteracyjna

xi = -2
yi = -1

for i in range(1000):
    xi, yi = (yi + 3*xi*xi)/(5*yi), -xi*xi -5 *xi - 7

print('Iteracyjna', xi, yi)

# Newton-Raphson 1

x = -4
y = -2

x, y = newton_raphson(x, y)

print('Newton Raphson 1', x, y)

# Newton-Raphson 2

x = 0
y = 0

x, y = newton_raphson(x, y)

print('Newton Raphson 2', x, y)

# Newton-Raphson 3

x = 10
y = 0

x, y = newton_raphson(x, y)

print('Newton Raphson 3', x, y)

plt.show()
