import numpy as np
from matplotlib import pyplot as plt


def dx_dt(xx, yy, zz):
    return -10*xx + 10*yy


def dy_dt(xx, yy, zz):
    return 28*xx - yy - xx*zz


def dz_dt(xx, yy, zz):
    return -8/3*zz + xx*yy


t0 = 0
tk = 25
h = 0.03125

x = 5
y = 5
z = 5

f_x = np.array([x])
f_y = np.array([y])
f_z = np.array([z])

t_vector = np.arange(t0, tk, h)
for t_i in t_vector:
    k1x = dx_dt(x, y, z)
    k1y = dy_dt(x, y, z)
    k1z = dz_dt(x, y, z)

    fxh = x + k1x * h / 2
    fyh = y + k1y * h / 2
    fzh = z + k1z * h / 2

    k2x = dx_dt(fxh, fyh, fzh)
    k2y = dy_dt(fxh, fyh, fzh)
    k2z = dz_dt(fxh, fyh, fzh)

    fxh = x + k2x * h / 2
    fyh = y + k2y * h / 2
    fzh = z + k2z * h / 2

    k3x = dx_dt(fxh, fyh, fzh)
    k3y = dy_dt(fxh, fyh, fzh)
    k3z = dz_dt(fxh, fyh, fzh)

    xk = x + k3x * h
    yk = y + k3y * h
    zk = z + k3z * h

    k4x = dx_dt(xk, yk, zk)
    k4y = dy_dt(xk, yk, zk)
    k4z = dz_dt(xk, yk, zk)

    x = x + 1/6*(k1x + 2*k2x + 2*k3x + k4x)*h
    y = y + 1/6*(k1y + 2*k2y + 2*k3y + k4y)*h
    z = z + 1/6*(k1z + 2*k2z + 2*k3z + k4z)*h

    f_x = np.append(f_x, x)
    f_y = np.append(f_y, y)
    f_z = np.append(f_z, z)

f_x = f_x[:-1]
f_y = f_y[:-1]
f_z = f_z[:-1]
fig1, ax2 = plt.subplots()

plt.plot(t_vector, f_x, label='fx(t)')
plt.plot(t_vector, f_y, label='fy(t)')
plt.plot(t_vector, f_z, label='fz(t)')
plt.legend()

fig2, ax2 = plt.subplots()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot(f_x, f_y, f_z)
plt.show()


