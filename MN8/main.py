import numpy as np
from matplotlib import pyplot as plt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = np.loadtxt('data11.txt')
    # print(data)
    data_x = np.transpose(data[:, 0])
    data_y = np.transpose(data[:, 1])

    array_of_fx = data_y
    b_array = [data_y[0]]
    list_of_next = np.array((array_of_fx[1:] - array_of_fx[:-1]) / (data_x[1:] - data_x[:-1]))
    array_of_fx = list_of_next

    i = 1

    # add
    b_array.append(list_of_next[0])
    while(len(list_of_next) > 1):
        list_of_next = np.array((array_of_fx[1:] - array_of_fx[:-1]) / (data_x[1 + i:] - data_x[:-1 - i]))
        array_of_fx = list_of_next
        i += 1

        b_array.append(list_of_next[0])

    step = 0.001
    x_vector, y_vector_newton = np.arange(data_x.min(), data_x.max() + step, step), np.array([])

    # calculate polynomial newton
    for x in x_vector:
        pol = b_array[len(b_array) - 1]
        for i in range(len(b_array) - 2, -1, -1):
            pol = pol * (x - data_x[i]) + b_array[i]
        y_vector_newton = np.append(y_vector_newton, pol)

    # sklejane

    h = data_x[-1] - data_x[-2]

    H = np.zeros((len(data_x), len(data_x)))
    for i in range(len(data_x)):
        for j in range(len(data_x)):
            if (i == 0 and j == 0) or (
                    i == len(data_x) - 1 and j == len(data_x) - 1):
                H[i][j] = 1
            elif i == j:
                H[i][j] = 2 * (h + h)
            elif (i == 0 and j == 1) or (i == len(data_x) - 1 and j == len(data_x) - 2):
                pass
            elif (i == j + 1) or (j == i + 1):
                H[i][j] = h
            else:
                pass

    F = np.zeros((len(data_x), 1))
    for i in range(1, len(data_x) - 1):
        f_xn_01 = data_y[i + 1] - data_y[i]
        xn_01 = data_x[i + 1] - data_x[i]
        f_xn_12 = data_y[i] - data_y[i - 1]
        xn_12 = data_x[i] - data_x[i - 1]
        F[i] = 3 * (f_xn_01 / xn_01 - f_xn_12 / xn_12)

    ci = np.linalg.solve(H, F)

    ci = ci.reshape(len(data_x))

    ai = data_y[0:-1]
    bi = (data_y[1:] - data_y[0:-1])/h - (h/3)*(2*ci[0:-1] + ci[1:])
    di = (ci[1:] - ci[0:-1])/(3*h)

    x_result = []
    result = []

    for i in range(len(data_x) - 1):
        for x in np.arange((h * i) + min(data_x), (h * i) + (min(data_x) + h), step):
            x_result.append(x)
            y = ai[i] + bi[i] * (x - data_x[i]) + ci[i] * ((x - data_x[i]) ** 2) + di[i] * (
                        (x - data_x[i]) ** 3)
            result.append(y)

    fig, ax = plt.subplots()
    plt.plot(x_vector, y_vector_newton, label='Newton')
    plt.plot(x_result, result, label='Sklejane')
    plt.plot(data_x, data_y, marker='o', linestyle='none', label='Punkty')
    plt.legend()
    plt.show()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
