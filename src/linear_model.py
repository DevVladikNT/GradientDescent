import numpy as np
import matplotlib.pyplot as plt

import loss_surface

k = 0.01
steps = 400

a = np.random.random(2)
a = a.reshape(-1, 1)


def loss(x, y):
    """MSE loss function"""
    result = x.dot(a) - y
    result = result ** 2
    return np.sum(result) / len(x)


def gradient(x, y):
    """
    Calculates gradient vector of loss function for X and Y
    :param x: matrix (n, poly + 1)
    :param y: matrix (n, 1)
    :return: gradient vector with size n
    """
    # Init partial derivatives
    df_da0 = 0.0
    df_da1 = 0.0
    for i in range(len(x)):
        df_da0 += 2 * (a[0] * x[i, 0] + a[1] * x[i, 1] - y[i, 0]) * x[i, 0]
        df_da1 += 2 * (a[0] * x[i, 0] + a[1] * x[i, 1] - y[i, 0]) * x[i, 1]
    # "/ len(x)" only decreases gradient vector's modulus
    return np.array([df_da0, df_da1]).reshape(-1, 1) / len(x)


def main():
    global a
    # x, y = data_gen.generate()
    x, y = np.array([[1, -8], [1, 2]]), np.array([[6], [11]])

    counter = 0
    a_array = []  # Array of several coefficients for plot
    for i in range(steps):
        a_array.append(a)
        if i % (steps/5) == 0:
            print(f'{a.reshape(-1)}: {loss(x, y)}')
            plt.plot(x[:, 1], np.sum(x.dot(a), axis=1), label=f'Step {i}')
            counter += 1
        a = a - gradient(x, y) * k

    plt.plot(x[:, 1], y[:, 0], 'o')
    plt.title('Gradient Descent')
    plt.legend()
    plt.show()
    loss_surface.plot_grad_desc(a_array)


if __name__ == '__main__':
    main()
