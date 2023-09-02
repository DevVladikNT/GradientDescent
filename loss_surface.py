import numpy as np
import matplotlib.pyplot as plt


def plot_grad_desc(a_array):
    # Surface of loss function in the model parameter space
    a1, a0 = np.meshgrid(np.arange(-0.5, 1, 0.001), np.arange(-5, 15, 0.01))
    z = (-8 * a1 + 1 * a0 - 6) ** 2 + (2 * a1 + 1 * a0 - 11) ** 2

    # Linear function which is solution for:
    # x = [1, -8], y = [6]
    a0_l1 = np.array([2, 10, 14])
    a1_l1 = np.array([-0.5, 0.5, 1])
    z_l1 = np.array([0, 0, 0])

    # Linear function which is solution for:
    # x = [1, 2], y = [11]
    a0_l2 = np.array([12, 10, 9])
    a1_l2 = np.array([-0.5, 0.5, 1])
    z_l2 = np.array([0, 0, 0])

    # Line which shows how our loss changes
    a0_gd = []
    a1_gd = []
    z_gd = []
    for row in a_array:
        a0_gd.append(row[0])
        a1_gd.append(row[1])
        z_gd.append((-8 * row[1] + 1 * row[0] - 6) ** 2 + (2 * row[1] + 1 * row[0] - 11) ** 2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(a1, a0, z, alpha=0.2)
    ax.plot(a1_l1, a0_l1, z_l1, label='Solution for first object')
    ax.plot(a1_l2, a0_l2, z_l2, label='Solution for second object')
    ax.plot(a1_gd, a0_gd, z_gd, label='Loss changes')
    plt.title('Loss surface')
    plt.legend(loc='lower left')
    plt.show()
