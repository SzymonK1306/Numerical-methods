import numpy as np
from matplotlib import pyplot as plt


# rownanie roznicowe
# G(z) = (z^2 + z + 1)/(z^3 - 3,4z^2 + 3,25z - 0,9)

def step_response(AA, BB, CC, xx, yy):
    # odpowiedz skokowa
    for ui in u:
        x_n = AA @ xx + BB * ui
        y_n = CC @ xx
        yy.append(y_n[0])
        xx = x_n

    return yy

# matrices
A = np.array([[3.4, -3.25, 0.9],
             [1, 0, 0],
             [0, 1, 0]])

B = np.array([[1], [0], [0]])

C = np.array([[1, 1, 1]])

# number of samples
size = 100

# initial condition vector
x = np.array([[0], [0], [0]])

# input vector
u = np.ones(size)

# time vector
t = np.arange(0, size, 1)

x_vector = [x]

# output vector of object (without regulator)
y = []

# calculate step response
y = step_response(A, B, C, x, y)        # unstable

# LQR controller
c1 = 10000
c2 = 10000
Q = np.eye(3) * c1
R = c2

# calculate P
P = np.zeros((3, 3))
for i in range(100):
    P_next = Q + np.transpose(A) @ (
            P - P @ B @ np.linalg.inv(R + np.transpose(B) @ P @ B) @ np.transpose(B) @ P) @ A
    P = P_next

print(P)

# calculate F
F = np.linalg.inv(R + np.transpose(B) @ P @ B) @ np.transpose(B) @ P @ A
print(F)

# new object with controller
A_new = A - B @ F
x_new = np.array([[0], [0], [0]])
y_new = []
uk_vector = np.zeros(size)

for n in range(size):
    x_n = A_new @ x_new + B * 1
    y_n = C @ x_new
    uk_vector[n] = -F @ x_new
    y_new.append(y_n[0])
    x_new = x_n

# plot
y = np.array(y)
y_new = np.array(y_new)
uk_vector = np.array(uk_vector)


fig, ax = plt.subplots(2)
fig.suptitle('LQR controller')

ax[0].set_title('Response of object without controller - unstable')
ax[1].set_title('Response of object with LQR controller')

ax[0].set(xlabel='time [n]', ylabel='y[n]')
ax[1].set(xlabel='time [n]', ylabel='y[n]')

ax[0].plot(t, y, 'r.', label='y[n]')
ax[1].plot(t, y_new, 'g.', label='y[n]')
ax[1].plot(t, uk_vector, 'b.', label='u[n]')

ax[0].legend(loc='upper left')
ax[1].legend(loc='upper left')

print('Stałe c1 i c2 wpływają na położenie biegunów układu. Oznacza to, że wpływamy na wartość przeregulowania, czasu ustalania, wzmocnienia statycznego')

plt.show()

