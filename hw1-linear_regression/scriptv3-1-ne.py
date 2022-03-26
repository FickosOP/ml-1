import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import pandas as pd
import sys


def load_set(path):
    df = pd.read_csv(path, sep=",", index_col=False)
    df.columns = ["x", "y"]
    data = np.array(df, dtype=float)

    return np.array([data[:, 0]]).T, np.array([data[:, 1]]).T


def normal_equation(x, y):
    m = x.shape[0]

    x = np.append(x, np.ones((m, 1)), axis=1)

    y = y.reshape(m, 1)

    theta = np.dot(inv(np.dot(x.T, x)), np.dot(x.T, y))

    print(theta)
    return theta


def fit(x, y):
    m, c = normal_equation(x, y)
    return m, c


def predict(x, m, c):
    return h(x, m, c)


def calculate_rmse(predicted, true):
    return np.sqrt(((predicted - true) ** 2).mean())


def plot_line(m, c, x, y):
    l_space = np.linspace(0, 0.5, 100)
    func = m * l_space + c

    plt.plot(l_space, func, '-r')
    plt.title(f"Graph on training set")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x, y)
    plt.show()


def main(train_path, test_path):
    x, y = load_set(train_path)

    m, c = fit(x, y)
    # print(f"{m} x + {c}")

    # plot_line(m, c, x, y)

    x_test, y_true = load_set(test_path)

    y_predicted = predict(x_test, m, c)

    # plot_line(m, c, x_test, y_true)

    error = calculate_rmse(y_predicted, y_true)

    print(error)


if __name__ == '__main__':
    main("data/train.csv", "data/test_preview.csv")
