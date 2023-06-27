import numpy as np
from matplotlib import pyplot as plt

def rozwiniecie(n, x):
    range = np.arange(1, n + 1)
    elements = (x**range)/range
    result = -np.sum(elements)
    return result


number_of_elements = 20
x = 0.5
series_value = rozwiniecie(number_of_elements, x)
print(series_value)
original_value = np.log(1 - x)

x_range = np.linspace(0, 0.999, 1000)
original_list = np.log(1-x_range)
first = []
second = []
third = []
print("{:<3} {:<5} {:<15} {:<15}".format('n', 'x', 'Absolute error', 'Relative error'))
for nn in range(number_of_elements):
    error = original_value - rozwiniecie(nn, x)
    print("{:<3} {:<5} {:<15} {:<15}".format(nn, x, np.round_(error, 10), error / original_value))

for xx in x_range:
    original_list
    first.append(rozwiniecie(1, xx))
    second.append((rozwiniecie(10, xx)))
    third.append(rozwiniecie(50, xx))

fig, ax = plt.subplots()
ax.plot(x_range, original_list, label='Original')
ax.plot(x_range, first, label='n = 1')
ax.plot(x_range, second, label='n = 10')
ax.plot(x_range, third, label='n = 50')
plt.legend()
plt.show()