from matplotlib import pyplot as plt
import numpy as np
from scipy.linalg import lu


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    Qa = 200
    Qb = 300
    Qc = 150
    Qd = 350

    ca = 2
    cb = 2

    E12 = 25
    E23 = 50
    E34 = 50
    E35 = 25

    Ws = 1500
    Wg = 2500

    # first calculation
    A = np.array([[-Qa-E12, E12, 0, 0, 0],
                  [E12+Qa, -Qa-Qb-E12-E23, E23, 0, 0],
                  [0, Qa+Qb+E23, -Qa-Qb-E23-E35-E34, E34, E35],
                  [0, 0, Qa+Qb-Qd+E34, -Qc-E34, 0],
                  [0, 0, Qa+Qb-Qc+E35, 0, -Qd-E35]])

    B = np.array([[-Ws-Qa*ca], [-Qb*cb], [0], [0], [-Wg]])

    p, l, u = lu(A)

    d = np.linalg.inv(l)@B
    x = np.linalg.inv(u)@d
    print('X vector', '\n', x)

    # change Wg and Ws
    Ws1 = 800
    Wg1 = 1200
    B1 = np.array([[-Ws1 - Qa * ca], [-Qb * cb], [0], [0], [-Wg1]])

    d1 = np.linalg.inv(l)@B1
    x1 = np.linalg.inv(u)@d1
    print('X vector after reduction', '\n', x1)

    A_inv = np.zeros((5, 5))

    # A-1
    diag = np.eye(5, dtype=float)

    for i in range(5):
        dd = np.linalg.inv(l)@diag[i]
        xx = np.linalg.inv(u)@dd
        A_inv[i] = xx

    A_inv = np.transpose(A_inv)
    print('Iverse matrix', '\n', A_inv)

    percentage_grill = 100 * np.abs(A_inv[3][4] * Wg / x[3])
    percentage_smokers = 100 * np.abs(A_inv[3][0] * Ws / x[3])
    percentage_street = 100 * np.abs((A_inv[3][0] * Qa*ca + A_inv[3][1] * Qb*cb) / x[3])

    print('Grill percentage', percentage_grill[0], '%')
    print('Smokers percentage', percentage_smokers[0], '%')
    print('street percentage', percentage_street[0], '%')


