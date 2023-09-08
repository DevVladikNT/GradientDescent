import numpy as np
import matplotlib.pyplot as plt

import data_gen

n = 20
poly = 2
max_val = 10
k = 0.001  # Increase k for higher degrees than 2
steps = 1000

a = 1 / (np.random.random(poly + 1) + 0.2 * (np.arange(poly + 1) + 1))
a = a / (np.arange(poly + 1) + 1)
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
    # Vector of partial derivatives
    df_da = np.zeros(poly + 1)
    # For each partial derivative
    for i in range(poly + 1):
        # For each x
        for j in range(len(x)):
            value = 0
            for p in range(poly + 1):
                value += a[p] * x[j, p]
            df_da[i] = df_da[i] + (value - y[j, 0]) * x[j, i]
    return df_da.reshape(-1, 1) / len(x)


def main():
    global a
    x, y = data_gen.generate(n=n, poly=poly, max_val=max_val)

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


if __name__ == '__main__':
    main()
