import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import sys


def load_set(path):
    df = pd.read_csv(path, sep=",", index_col=False)
    df.columns = ["x", "y"]

    df.hist(grid=False,
            figsize=(10, 6),
            bins=30)
    plt.show()

    df.insert(len(df.columns), 'Y_Sqrt',
              np.sqrt(df.iloc[:, 1]))
    df.insert(len(df.columns), 'Y_log',
              np.log(df.iloc[:, 1]))
    df.insert(len(df.columns), 'Y_Boxcox',
              boxcox(df.iloc[:, 1])[0])
    print(f"BEST LAMBDA: {boxcox(df.iloc[:, 1])[1]}")
    df.hist(grid=False,
            figsize=(10, 6),
            bins=30)
    plt.show()

    print(df.agg(['skew', 'kurtosis']).transpose())

    lmbda = boxcox(df.iloc[:, 1])[1]

    data = np.array(df, dtype=float)

    return np.array(data[:, 0]), np.array(data[:, 1]), np.array(data[:, -1]), lmbda


def cost_function(m, c, x, y):
    return sum((h(x, m, c) - y) ** 2)/(len(x))


def h(x, m, c):
    return m*x + c


def plot_error_function(j, i):
    plt.scatter(j, i)
    plt.show()


def gradient_descent(x, y, learning_rate=0.1, max_iterations=1000, stopping_threshold=1e-9):

    j_all = []
    iterations = []
    m = 1
    c = 0
    previous_cost = None
    for _ in range(max_iterations):
        current_cost = cost_function(m, c, x, y)
        j_all.append(current_cost)
        iterations.append(m)
        if previous_cost and previous_cost - current_cost < stopping_threshold:
            print("Breaking from for loop...")
            break
        previous_cost = current_cost

        h_x = h(x, m, c)
        m = m - (learning_rate * sum((h_x - y) * x)/len(x))
        c = c - (learning_rate * sum((h_x - y))/len(x))

    # plot_error_function(j_all, iterations)
    return m, c


def create_new_sets(x_parts, y_parts, y_bc_parts, i):
    new_x, new_y, new_x_test, new_y_true = [], [], [], []

    for j in range(len(x_parts)):
        if j == i:
            for k in range(len(x_parts[j])):
                new_x_test.append(x_parts[j][k])
                new_y_true.append(y_parts[j][k])
        else:
            for k in range(len(x_parts[j])):
                new_x.append(x_parts[j][k])
                new_y.append(y_bc_parts[j][k])
    return np.array(new_x), np.array(new_y), np.array(new_x_test), np.array(new_y_true)


def fit(x, y, y_bc, lmbda):
    # apply cross validation
    x_parts = np.array_split(x, 5)
    y_parts = np.array_split(y, 5)
    y_bc_parts = np.array_split(y_bc, 5)
    error_list = []
    parameter_list = []
    for i in range(len(x_parts)):
        new_x, new_y, new_x_test, new_y_true = create_new_sets(x_parts, y_parts, y_bc_parts, i)
        m, c = gradient_descent(new_x, new_y, learning_rate=0.1, max_iterations=10000)
        # plot_line(m, c, new_x_test, new_y_true, lmbda)
        parameter_list.append((m, c))
        y_predicted = predict(new_x_test, m, c, lmbda)
        print(f"True: {new_y_true}\n")
        err = calculate_rmse(y_predicted, new_y_true)
        error_list.append(err)
        # wk = input()
    print(error_list)
    print(f"Average error: {sum(error_list)/len(error_list)}")
    print(f"Lowest error: {min(error_list)}")
    print(f"All parameter combinations: {parameter_list}")
    print(f"Best parameter combination: {parameter_list[error_list.index(min(error_list))]}")

    # m, c = gradient_descent(x, y, learning_rate=0.1, max_iterations=10000)
    return parameter_list[error_list.index(min(error_list))]


def predict(x, m, c, lmbda):
    reverted = inv_boxcox(h(x, m, c), lmbda)
    print(f"Predicted:{reverted}\n")
    return reverted


def calculate_rmse(predicted, true):
    return np.sqrt(((predicted - true) ** 2).mean())


def plot_line(m, c, x, y, lmbda):
    l_space = np.linspace(0, 0.5, 100)
    func = inv_boxcox(h(l_space, m, c), lmbda)

    m = 114.25301007
    c = -21.85025345
    func_prob = m * l_space + c

    plt.plot(l_space, func, '-r')
    plt.plot(l_space, func_prob, '-g')
    plt.title(f"Graph on training set")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x, y)
    plt.show()


def remove_outliers(x, y):
    new_x, new_y, new_y_bc = [], [], []
    for i in range(len(x)):
        if x[i] < 0.4 and y[i] > 35:
            continue

        new_x.append(x[i])
        new_y.append(y[i])
    print(f"Outliers: {len(x) - len(new_x)}")
    return np.array(new_x), np.array(new_y)


def main(train_path, test_path):
    x, y, y_bc, lmbda = load_set(train_path)

    plt.scatter(x, y)
    plt.show()

    m, c = fit(x, y, y_bc, lmbda)
    print(f"{m} x + {c}")

    x_test, y_true, y_bc, lmbda = load_set(test_path)

    y_predicted = predict(x_test, m, c, lmbda)

    plot_line(m, c, x_test, y_true, lmbda)

    error = calculate_rmse(y_predicted, [4.46, 7.16, 0.2, 0.93])

    print(error)


if __name__ == '__main__':
    main("data/train.csv", "data/test_preview.csv")
