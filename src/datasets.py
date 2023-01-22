from sys import path
from sklearn.datasets import make_blobs, make_circles
import matplotlib.pyplot as plt
import numpy as np


class Datasets:
    @staticmethod
    def get_random_blobs_set(n_samples=100) -> tuple:
        X, y = make_blobs(n_samples=n_samples, n_features=2, centers=2, random_state=0)
        y = y.reshape((y.shape[0], 1))

        return (X, y)

    @staticmethod
    def get_random_circles_set(n_samples=100) -> tuple:
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.3, random_state=0)
        X = X.T
        y = y.reshape((1, y.shape[0]))

        return (X, y)

    @staticmethod
    def display_circles_set(X, y) -> None:
        plt.scatter(X[0, :], X[1, :], c=y, cmap="summer")
        plt.show()

    @staticmethod
    def display_blobs_set(X, y) -> None:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="summer")
        plt.show()
