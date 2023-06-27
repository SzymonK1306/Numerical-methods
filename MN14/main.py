import numpy as np
from matplotlib import pyplot as plt


h = 0.05
Ta = 100
dx = 0.5
n = 19

A = np.zeros((361, 361))
B = np.zeros((361, 1))

Ttop = np.arange(300, 405, 5)
Ttop = Ttop[::-1]

Tleft = Ttop

Tbot = np.arange(200, 305, 5)
Tbot = Tbot[::-1]

Tright = Tbot

mat_ot = np.zeros((21, 21))
mat_ot[:, 0] = Tleft
mat_ot[0, :] = Ttop
mat_ot[:, 20] = Tright
mat_ot[20, :] = Tbot

for i in range(n):
    for j in range(n):
        l, m = i + 1, j + 1
        B[n*i + j] = (dx * dx) * h * Ta + mat_ot[l + 1, m] + mat_ot[l - 1, m] + mat_ot[l, m + 1] + mat_ot[l, m - 1]

        A[n*i + j, n*i + j] = 4 + (dx * dx) * h
        if j + 1 < n:
            A[n*i + j, (n * i + j) + 1] = -1
        if j - 1 >= 0:
            A[n*i + j, (n * i + j) - 1] = -1
        if i + 1 < n:
            A[n*i + j, n * (i + 1) + j] = -1
        if i - 1 >= 0:
            A[n*i + j, n * (i - 1) + j] = -1
# solve
res = np.linalg.solve(A, B)

print(np.min(res))
res = res.reshape(n, n)



rows, columns = mat_ot.shape

# Step 3
inner_region = mat_ot[1:rows-1, 1:columns-1]

# Step 4
inner_region[:,:] = res

x, y = np.arange(mat_ot.shape[1]), np.arange(mat_ot.shape[0])
X, Y = np.meshgrid(x + 1, y + 1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, mat_ot, cmap='viridis')
plt.show()

