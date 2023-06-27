import numpy as np
from scipy.integrate import trapz

def function(x):
    x_power = np.array([(x)**xx for xx in range(0, 5)])
    return np.sum(f_x*x_power)


a = -9
b = 5

f_x = np.array([-0.08675, -0.5421, 0.8672, -2.31, 3.673])
f_x = np.flip(f_x)
# analityczna
coefficients = np.array([0.2*-0.08675, 0.25*-0.5421, 0.8672/3, -2.31/2, 3.673])
coefficients = np.flip(coefficients)
x_9_factor = np.array([(-9)**x for x in range(1, 6)])
x_5_factor = np.array([5**x for x in range(1, 6)])

integral = np.sum(x_5_factor*coefficients) - np.sum(x_9_factor*coefficients)

print('Analityczna', integral)

def fun_gauss(x):
    result = f_x[4] * ((b + a) / 2 + x * (b - a) / 2)**4 \
             +  f_x[3] * ((b + a) / 2 + x * (b - a) / 2)**3 \
             + f_x[2] * ((b + a) / 2 + x * (b - a) / 2)**2 \
             + f_x[1] * ((b + a) / 2 + x * (b - a) / 2) \
             +  f_x[0]
    return result * (b - a) / 2

# def fun(x):
#     return (-0.06452) * x ** 4 + 0.5432 * x ** 3 - 0.7523 * x ** 2 - 3.132 * x + 2.756
# Gauss
c_gauss = [5/9, 8/9, 5/9]
x_gauss = [-np.sqrt(3/5), 0, np.sqrt(3/5)]

to_sum = [c_gauss[i] * fun_gauss(x_gauss[i]) for i in range(3)]

result_gauss = np.sum(to_sum)


#  Romberg
romberg_iterations = []
h = (b - a)

romberg_iterations.append([(h / 2) * (function(a) + function(b))])

for i in range(1, 16):
    h = h/2
    sum = 0

    for k in range(1, 2**i, 2):
        sum = sum + function(a + k*h)

    row_i = [0.5 * romberg_iterations[i - 1][0] + sum * h]

    for j in range(1, i+1):
        r_ij = row_i[j-1] + (row_i[j-1] - romberg_iterations[i - 1][j - 1]) / (4. ** j - 1)
        row_i.append(r_ij)

    romberg_iterations.append(row_i)

    error = np.abs((100 * (romberg_iterations[-1][-1] - romberg_iterations[-1][-2])) / (romberg_iterations[-1][-1]))
    if error < 0.2:
        break

for i, romberg_result in enumerate(romberg_iterations):
    print(f'Romberg iteration {i+1}', romberg_result[-1])

print('Gauss', result_gauss)
# Table


