import numpy as np
from matplotlib import pyplot as plt

tk = int(input('Podaj tk: '))
n = 20

t_vector = np.arange(0, tk, tk/n)
t_analityczny = np.arange(0, tk, 0.001)

# dy_dt = (3t - 4t^2)*sqrt(y)
# y =
# analityczna
c = 36/144
# y_a = np.square(-2 / 3 * t_analityczny ** 3 + 3 / 4 * t_analityczny ** 2 + 1)
y_a = 1/144*np.square(-8*t_analityczny**3 + 9*t_analityczny**2 + 12)


def f(t, y):
    return (3*t - 4*t**2) * np.sqrt(y)


# Eulera
h = tk/n
y_prev = 1
y_euler = np.array([y_prev])
for t_i in t_vector:
    y_prev = y_prev + h * f(t_i, y_prev)
    y_euler = np.append(y_euler, [y_prev])

y_euler = y_euler[:-1]
print(y_euler[10])

# Heuna
y_prev = 1
y_heun = np.array([y_prev])
for t_i in t_vector:
    k1 = f(t_i, y_prev)
    k2 = f(t_i + h, y_prev + k1*h)

    y_prev = y_prev + h*(k1 + k2)/2
    y_heun = np.append(y_heun, y_prev)

y_heun = y_heun[:-1]

# środkowy
y_prev = 1
y_mid = np.array([y_prev])
for t_i in t_vector:
    k1 = f(t_i, y_prev)
    k2 = f(t_i + h/2, y_prev + k1*h/2)

    y_prev = y_prev + h*k2
    y_mid = np.append(y_mid, y_prev)

y_mid = y_mid[:-1]

fig, ax = plt.subplots()
plt.plot(t_analityczny, y_a, label='analitycznie')
plt.plot(t_vector, y_euler, label='Euler')
plt.plot(t_vector, y_heun, label='Heun')
plt.plot(t_vector, y_mid, label='Punkt środkowy')
plt.legend()
plt.show()
