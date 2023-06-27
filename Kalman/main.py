from matplotlib import pyplot as plt
import numpy as np

data = np.loadtxt("measurements11.txt")

T = 1

px = data[:, 0]
py = data[:, 1]

F = np.array([[1, 0, T, 0],
             [0, 1, 0, T],
             [0, 0, 1, 0],
             [0, 0, 0, 1]])
G = np.array([[0, 0],
              [0, 0],
              [1, 0],
              [0, 1]])
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])
# state
s = np.array([[px[0], py[0], 0, 0]]).transpose()
# Prediction covariance matrix (initial)
P = np.array([[5, 0, 0, 0],
             [0, 5, 0, 0],
             [0, 0, 5, 0],
             [0, 0, 0, 5]])
# Process noise covariance matrix
Q = np.array([[0.25, 0],
             [0, 0.25]])
# Measurement noise covariance matrix
R = np.array([[2, 0],
             [0, 2]])

# lists
trajectory = []
estimation = []

for i in range(len(px) - 1):
    # PREDICTION
    sn1 = F @ s
    Pn1 = F @ P @ np.transpose(F) + G @ Q @ np.transpose(G)
    zn1 = H @ sn1

    # MEASUREMENT SENSOR
    # innovation
    en1 = np.array([[px[i + 1]], [py[i + 1]]]) - zn1
    # covariance innovation matrix
    Sn1 = H @ Pn1 @ np.transpose(H) + R
    # gain
    Kn1 = Pn1 @ np.transpose(H) @ np.linalg.inv(Sn1)

    # ACTUALISATION OF STATE ESTIMATION
    sn1n1 = sn1 + Kn1 @ en1
    Pn1n1 = (np.eye(4) - Kn1 @ H) @ Pn1

    # new cycle prepare
    trajectory.append(sn1n1)
    s = sn1n1
    P = Pn1n1

estimation.append(s)

# Estimation
for i in range(5):
    sn1 = F @ s
    # Pn1 = F @ P @ np.transpose(F) + G @ Q @ np.transpose(G)
    estimation.append(sn1)

    s = sn1
    # P = Pn1

trajectory = np.array(trajectory)
estimation = np.array(estimation)

fig, ax = plt.subplots()
plt.title('Kalman filter')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(px, py, marker='x', color='r', markersize=5, linestyle='none', label='Measurements')
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b', label='Trajectory')
plt.plot(estimation[:, 0], estimation[:, 1], 'b', linestyle='dotted', label='Prediction')
plt.legend()
plt.show()
