import unittest
import numpy as np

from polynomial_model import loss, gradient


class TestModel(unittest.TestCase):
    def test_loss(self):
        a = np.array([0, 0]).reshape(-1, 1)
        x = np.array([[1, 1],
                      [1, 2]])
        y = np.array([2, 3]).reshape(-1, 1)
        self.assertEqual(loss(x, y, params=a), 6.5)

    def test_loss_zero(self):
        a = np.array([1, 1]).reshape(-1, 1)
        x = np.array([[1, 1],
                      [1, 2]])
        y = np.array([2, 3]).reshape(-1, 1)
        self.assertEqual(loss(x, y, params=a), 0)

    def test_gradient(self):
        a = np.array([0, 0]).reshape(-1, 1)
        x = np.array([[1, 1],
                      [1, 2]])
        y = np.array([2, 3]).reshape(-1, 1)
        # gradient function should return [[-2.5], [-4]], so, sum is 6.5
        self.assertEqual(np.sum(gradient(x, y, params=a), axis=0), -6.5)

    def test_gradient_zero(self):
        a = np.array([1, 1]).reshape(-1, 1)
        x = np.array([[1, 1],
                      [1, 2]])
        y = np.array([2, 3]).reshape(-1, 1)
        # when loss=0, gradient is [[0], [0]]
        self.assertEqual(np.sum(gradient(x, y, params=a), axis=0), 0)


if __name__ == "__main__":
    unittest.main()
