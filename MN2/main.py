import numpy as np
import math
from matplotlib import pyplot as plt
from prettytable import PrettyTable

# log(1-x)
x = 0.5
true_derivative = -1/(1-x)
h_vector = np.geomspace(0.4, 0.4*math.pow(0.2, 19), num=20)
derivative_vector = (np.log(1-x-h_vector) - np.log(1-x+h_vector))/(2*h_vector)

error_vector = np.abs(true_derivative - derivative_vector)
table = PrettyTable()
table.add_column('h', h_vector)
table.add_column('derivative', derivative_vector)
table.add_column('Absolute error', error_vector)
print(table)

print(f'The smallest error for h = {h_vector[np.argmin(error_vector)]}')
fig, ax = plt.subplots()

ax.plot(h_vector, error_vector)
plt.title("(log(1-x)'")
plt.xlabel('h')
plt.ylabel('absolute error')
plt.yscale('log')
plt.xscale('log')
plt.show()
