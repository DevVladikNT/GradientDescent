import numpy as np


def generate(n: int = 20, poly: int = 1, max_val: float = 10):
    """
    Generates X and Y matrix with size (n, poly + 1) and (n, 1)
    which corresponds to a polynomial function of degree n
    :param n: sample size
    :param poly: degree of polynomial function
    :param max_val: max value of function argument
    :return: X, Y
    """
    # Generates n values from 0 to max_val
    x = np.arange(n) * max_val / n
    x = x.reshape(-1, 1)

    if poly == 1:
        # For demonstration in readme
        a = np.array([10, 0.5])
        x = x * np.array([1, 1])
        x[:, 0] = 1
    else:
        # Random coefficients for polynomial function
        # (smaller values for higher degrees)
        a = 1 / (np.random.random(poly + 1) + 0.2 * (np.arange(poly + 1) + 1))
        a = a / (np.arange(poly + 1) + 1)
        print(f'a: {a}')
        # Make matrix from x vector
        x = x * np.array([1] * (poly + 1))
        for i in range(poly + 1):
            x[:, i] = x[:, i] ** i

    # Calculation of function values
    y = x * a
    y = np.sum(y, axis=1)
    # Adding noise
    y = y + 10 * np.random.rand(len(y))
    y = y.reshape(-1, 1)

    return x, y
